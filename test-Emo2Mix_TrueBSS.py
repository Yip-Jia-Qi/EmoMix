#!/usr/bin/env/python3
"""Recipe for training a neural speech separation system on Libri2/3Mix datasets.
The system employs an encoder, a decoder, and a masking network.

To run this recipe, do the following:
> python train.py hparams/sepformer-libri2mix.yaml
> python train.py hparams/sepformer-libri3mix.yaml


The experiment file is flexible enough to support different neural
networks. By properly changing the parameter files, you can try
different architectures. The script supports both libri2mix and
libri3mix.


Authors
 * Cem Subakan 2020
 * Mirco Ravanelli 2020
 * Samuele Cornell 2020
 * Mirko Bronzi 2020
 * Jianyuan Zhong 2020
"""

import os
import sys
import torch
import torch.nn.functional as F
import torchaudio
import speechbrain as sb
import speechbrain.nnet.schedulers as schedulers
from speechbrain.utils.distributed import run_on_main
from torch.cuda.amp import autocast
from hyperpyyaml import load_hyperpyyaml
import numpy as np
from tqdm import tqdm
import csv
import logging

#packages for dataio
import itertools
from scipy.signal import resample_poly
from RAVDESS2Mix_BSS_prep import getEmotion, getIntensity
import pyloudnorm
import warnings
import random

# Define training procedure
class Separation(sb.Brain):
    def compute_forward(self, mix, targets, stage, noise=None):
        """Forward computations from the mixture to the separated signals."""

        # Unpack lists and put tensors in the right device
        mix, mix_lens = mix
        mix, mix_lens = mix.to(self.device), mix_lens.to(self.device)

        # Convert targets to tensor
        targets = torch.cat(
            [targets[i][0].unsqueeze(-1) for i in range(self.hparams.num_spks)],
            dim=-1,
        ).to(self.device)

        # Add speech distortions
        if stage == sb.Stage.TRAIN:
            with torch.no_grad():
                if self.hparams.use_speedperturb or self.hparams.use_rand_shift:
                    mix, targets = self.add_speed_perturb(targets, mix_lens)

                    mix = targets.sum(-1)

                    if self.hparams.use_wham_noise:
                        noise = noise.to(self.device)
                        len_noise = noise.shape[1]
                        len_mix = mix.shape[1]
                        min_len = min(len_noise, len_mix)

                        # add the noise
                        mix = mix[:, :min_len] + noise[:, :min_len]

                        # fix the length of targets also
                        targets = targets[:, :min_len, :]

                if self.hparams.use_wavedrop:
                    mix = self.hparams.wavedrop(mix, mix_lens)

                if self.hparams.limit_training_signal_len:
                    mix, targets = self.cut_signals(mix, targets)

        # Separation
        mix_w = self.hparams.Encoder(mix)
        est_mask = self.hparams.MaskNet(mix_w) #output [Batch, n_spk, Channel, Time]
        mix_w = torch.stack([mix_w] * self.hparams.num_spks)
        sep_h = mix_w * est_mask

        # Decoding
        est_source = torch.cat(
            [
                self.hparams.Decoder(sep_h[i]).unsqueeze(-1)
                for i in range(self.hparams.num_spks)
            ],
            dim=-1,
        )

        # T changed after conv1d in encoder, fix it here
        T_origin = mix.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]

        return est_source, targets

    def compute_objectives(self, predictions, targets):
        """Computes the si-snr loss"""
        return self.hparams.loss(targets, predictions)

    def fit_batch(self, batch):
        """Trains one batch"""
        # Unpacking batch list
        mixture = batch.mix_sig
        targets = [batch.s1_sig, batch.s2_sig]
        if self.hparams.use_wham_noise:
            noise = batch.noise_sig[0]
        else:
            noise = None

        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

        if self.auto_mix_prec:
            with autocast():
                predictions, targets = self.compute_forward(
                    mixture, targets, sb.Stage.TRAIN, noise
                )
                loss = self.compute_objectives(predictions, targets)

                # hard threshold the easy dataitems
                if self.hparams.threshold_byloss:
                    th = self.hparams.threshold
                    loss_to_keep = loss[loss > th]
                    if loss_to_keep.nelement() > 0:
                        loss = loss_to_keep.mean()
                else:
                    loss = loss.mean()

            if (
                loss < self.hparams.loss_upper_lim and loss.nelement() > 0
            ):  # the fix for computational problems
                self.scaler.scale(loss).backward()
                if self.hparams.clip_grad_norm >= 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.modules.parameters(), self.hparams.clip_grad_norm,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.nonfinite_count += 1

                logger.info(
                    "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                        self.nonfinite_count
                    )
                )
                if self.nonfinite_count>20:
                    raise Exception("infinite or empty loss happened too many times")
                
                loss.data = torch.tensor(0).to(self.device)
        else:
            predictions, targets = self.compute_forward(
                mixture, targets, sb.Stage.TRAIN, noise
            )
            loss = self.compute_objectives(predictions, targets)

            if self.hparams.threshold_byloss:
                th = self.hparams.threshold
                loss_to_keep = loss[loss > th]
                if loss_to_keep.nelement() > 0:
                    loss = loss_to_keep.mean()
            else:
                loss = loss.mean()

            if (
                loss < self.hparams.loss_upper_lim and loss.nelement() > 0
            ):  # the fix for computational problems
                loss.backward()
                if self.hparams.clip_grad_norm >= 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.modules.parameters(), self.hparams.clip_grad_norm
                    )
                self.optimizer.step()
            else:
                self.nonfinite_count += 1
                logger.info(
                    "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                        self.nonfinite_count
                    )
                )
                if self.nonfinite_count>20:
                    raise Exception("infinite or empty loss happened too many times")
                loss.data = torch.tensor(0).to(self.device)
        self.optimizer.zero_grad()

        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        snt_id = batch.id
        mixture = batch.mix_sig
        targets = [batch.s1_sig, batch.s2_sig]
        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

        with torch.no_grad():
            predictions, targets = self.compute_forward(mixture, targets, stage)
            loss = self.compute_objectives(predictions, targets)

        # Manage audio file saving
        if stage == sb.Stage.TEST and self.hparams.save_audio:
            if hasattr(self.hparams, "n_audio_to_save"):
                if self.hparams.n_audio_to_save > 0:
                    self.save_audio(snt_id[0], mixture, targets, predictions)
                    self.hparams.n_audio_to_save += -1
            else:
                self.save_audio(snt_id[0], mixture, targets, predictions)

        return loss.detach()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"si-snr": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            # Learning rate annealing
            if isinstance(
                self.hparams.lr_scheduler, schedulers.ReduceLROnPlateau
            ):
                current_lr, next_lr = self.hparams.lr_scheduler(
                    [self.optimizer], epoch, stage_loss
                )
                schedulers.update_learning_rate(self.optimizer, next_lr)
            else:
                # if we do not use the reducelronplateau, we do not change the lr
                current_lr = self.hparams.optimizer.optim.param_groups[0]["lr"]

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": current_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"si-snr": stage_stats["si-snr"]}, min_keys=["si-snr"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

    def add_speed_perturb(self, targets, targ_lens):
        """Adds speed perturbation and random_shift to the input signals"""

        min_len = -1
        recombine = False

        if self.hparams.use_speedperturb:
            # Performing speed change (independently on each source)
            new_targets = []
            recombine = True

            for i in range(targets.shape[-1]):
                new_target = self.hparams.speedperturb(
                    targets[:, :, i], targ_lens
                )
                new_targets.append(new_target)
                if i == 0:
                    min_len = new_target.shape[-1]
                else:
                    if new_target.shape[-1] < min_len:
                        min_len = new_target.shape[-1]

            if self.hparams.use_rand_shift:
                # Performing random_shift (independently on each source)
                recombine = True
                for i in range(targets.shape[-1]):
                    rand_shift = torch.randint(
                        self.hparams.min_shift, self.hparams.max_shift, (1,)
                    )
                    new_targets[i] = new_targets[i].to(self.device)
                    new_targets[i] = torch.roll(
                        new_targets[i], shifts=(rand_shift[0],), dims=1
                    )

            # Re-combination
            if recombine:
                if self.hparams.use_speedperturb:
                    targets = torch.zeros(
                        targets.shape[0],
                        min_len,
                        targets.shape[-1],
                        device=targets.device,
                        dtype=torch.float,
                    )
                for i, new_target in enumerate(new_targets):
                    targets[:, :, i] = new_targets[i][:, 0:min_len]

        mix = targets.sum(-1)
        return mix, targets

    def cut_signals(self, mixture, targets):
        """This function selects a random segment of a given length withing the mixture.
        The corresponding targets are selected accordingly"""
        randstart = torch.randint(
            0,
            1 + max(0, mixture.shape[1] - self.hparams.training_signal_len),
            (1,),
        ).item()
        targets = targets[
            :, randstart : randstart + self.hparams.training_signal_len, :
        ]
        mixture = mixture[
            :, randstart : randstart + self.hparams.training_signal_len
        ]
        return mixture, targets

    def reset_layer_recursively(self, layer):
        """Reinitializes the parameters of the neural networks"""
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        for child_layer in layer.modules():
            if layer != child_layer:
                self.reset_layer_recursively(child_layer)

    def save_results(self, test_data, save_name="test"):
        """This script computes the SDR and SI-SNR metrics and saves
        them into a csv file"""
        logger.info("Results on {}".format(save_name))
   
        # This package is required for SDR computation
        from mir_eval.separation import bss_eval_sources

        # Create folders where to store audio
        save_file = os.path.join(self.hparams.output_folder, f'{save_name}_results.csv')

        # Variable init
        all_sdrs = []
        all_sdrs_i = []
        all_sisnrs = []
        all_sisnrs_i = []
        csv_columns = ["snt_id", "sdr", "sdr_i", "si-snr", "si-snr_i"]


        test_loader = sb.dataio.dataloader.make_dataloader(
            test_data, **self.hparams.dataloader_opts
        )

        with open(save_file, "w") as results_csv:
            writer = csv.DictWriter(results_csv, fieldnames=csv_columns)
            writer.writeheader()

            # Loop over all test sentence
            with tqdm(test_loader, dynamic_ncols=True) as t:
                for i, batch in enumerate(t):

                    # Apply Separation
                    mixture, mix_len = batch.mix_sig
                    snt_id = batch.id
                    targets = [batch.s1_sig, batch.s2_sig]
                    if self.hparams.num_spks == 3:
                        targets.append(batch.s3_sig)

                    with torch.no_grad():
                        predictions, targets = self.compute_forward(
                            batch.mix_sig, targets, sb.Stage.TEST
                        )

                    # Compute SI-SNR
                    sisnr = self.compute_objectives(predictions, targets)

                    # Compute SI-SNR improvement
                    mixture_signal = torch.stack(
                        [mixture] * self.hparams.num_spks, dim=-1
                    )
                    mixture_signal = mixture_signal.to(targets.device)
                    sisnr_baseline = self.compute_objectives(
                        mixture_signal, targets
                    )
                    sisnr_i = sisnr - sisnr_baseline

                    # Compute SDR
                    sdr, _, _, _ = bss_eval_sources(
                        targets[0].t().cpu().numpy(),
                        predictions[0].t().detach().cpu().numpy(),
                    )

                    sdr_baseline, _, _, _ = bss_eval_sources(
                        targets[0].t().cpu().numpy(),
                        mixture_signal[0].t().detach().cpu().numpy(),
                    )

                    sdr_i = sdr.mean() - sdr_baseline.mean()

                    # Saving on a csv file
                    row = {
                        "snt_id": snt_id[0],
                        "sdr": sdr.mean(),
                        "sdr_i": sdr_i,
                        "si-snr": -sisnr.item(),
                        "si-snr_i": -sisnr_i.item(),
                    }
                    writer.writerow(row)

                    # Metric Accumulation
                    all_sdrs.append(sdr.mean())
                    all_sdrs_i.append(sdr_i.mean())
                    all_sisnrs.append(-sisnr.item())
                    all_sisnrs_i.append(-sisnr_i.item())

                row = {
                    "snt_id": "avg",
                    "sdr": np.array(all_sdrs).mean(),
                    "sdr_i": np.array(all_sdrs_i).mean(),
                    "si-snr": np.array(all_sisnrs).mean(),
                    "si-snr_i": np.array(all_sisnrs_i).mean(),
                }
                writer.writerow(row)
        
        #write to summary file
        summary_file = os.path.join(self.hparams.output_folder, f'summary_results.csv')
        summary_columns = ["emotion", "sdr", "sdr_i", "si-snr", "si-snr_i"]
        print(summary_file)
        with open(summary_file, "a") as summary_csv:
            writer = csv.DictWriter(summary_csv, fieldnames=summary_columns)
            # writer.writeheader()
            row = {
                "emotion": save_name,
                "sdr": np.array(all_sdrs).mean(),
                "sdr_i":np.array(all_sdrs_i).mean(),
                "si-snr": np.array(all_sisnrs).mean(),
                "si-snr_i":np.array(all_sisnrs_i).mean(),
            }
            writer.writerow(row)
            
        logger.info("Mean SISNR is {}".format(np.array(all_sisnrs).mean()))
        logger.info("Mean SISNRi is {}".format(np.array(all_sisnrs_i).mean()))
        logger.info("Mean SDR is {}".format(np.array(all_sdrs).mean()))
        logger.info("Mean SDRi is {}".format(np.array(all_sdrs_i).mean()))

    def save_audio(self, snt_id, mixture, targets, predictions):
        "saves the test audio (mixture, targets, and estimated sources) on disk"

        # Create outout folder
        save_path = os.path.join(self.hparams.save_folder, "audio_results")
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for ns in range(self.hparams.num_spks):

            # Estimated source
            signal = predictions[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, "item{}_source{}hat.wav".format(snt_id, ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

            # Original source
            signal = targets[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, "item{}_source{}.wav".format(snt_id, ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

        # Mixture
        signal = mixture[0][0, :]
        signal = signal / signal.abs().max()
        save_file = os.path.join(save_path, "item{}_mix.wav".format(snt_id))
        torchaudio.save(
            save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
        )

def dataio_prep(hparams):

    # emotion_list = [*range(1,9)]
    # emotion_combs = [*itertools.product(emotion_list,emotion_list)]
    # normal_emotions_list = [getEmotion(x[0])+"_"+getEmotion(x[1]) for x in emotion_combs]
    
    normal_emotions_list = hparams["normal_emotions_list"]

    # emotion_list = [*range(2,9)]
    # emotion_combs = [*itertools.product(emotion_list,emotion_list)]
    # strong_emotions_list = [getEmotion(x[0])+"_"+getEmotion(x[1]) for x in emotion_combs]
    
    strong_emotions_list = hparams["strong_emotions_list"]

    # 1. Define datasets
    normal_data = []
    for emotion in normal_emotions_list:
        normal_data.append(
            sb.dataio.dataset.DynamicItemDataset.from_csv(
                csv_path=f'{hparams["save_folder"]}/normal_normal/{emotion}.csv',                    
                )
            )
    strong_data = []
    for emotion in strong_emotions_list:
        strong_data.append(
            sb.dataio.dataset.DynamicItemDataset.from_csv(
                csv_path=f'{hparams["save_folder"]}/strong_strong/{emotion}.csv',                    
                )
            )
    @sb.utils.data_pipeline.takes("s1_wav","s2_wav")
    @sb.utils.data_pipeline.provides("mix_sig","s1_sig","s2_sig")
    def audio_pipeline_mix(s1_wav, s2_wav):
        """
        This audio pipeline defines the compute graph for dynamic mixing of the RAVDESS2Mix dataset
        Based on the original RAVDES2Mix mixing script, we downsample the audio before normalization 
        Function used is scipy.signal.resample_poly

        """
        rng = np.random.default_rng(9438)
        rgen = random.Random(7289)
        
        sources = []
        spk_files = [s1_wav,s2_wav]


        minlen = min(
            *[torchaudio.info(x).num_frames for x in spk_files],
        )

        meter = pyloudnorm.Meter(hparams["sample_rate"])

        MAX_AMP = 0.9
        MIN_LOUDNESS = -33
        MAX_LOUDNESS = -25

        def normalize(signal, is_noise=False):
            """
            This function normalizes the audio signals for loudness
            """
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c_loudness = meter.integrated_loudness(signal)
                if is_noise:
                    # target_loudness = random.uniform(
                    target_loudness = rgen.uniform(
                        MIN_LOUDNESS - 5, MAX_LOUDNESS - 5
                    )
                else:
                    # target_loudness = random.uniform(MIN_LOUDNESS, MAX_LOUDNESS)
                    target_loudness = rgen.uniform(MIN_LOUDNESS, MAX_LOUDNESS)
                signal = pyloudnorm.normalize.loudness(
                    signal, c_loudness, target_loudness
                )

                # check for clipping
                if np.max(np.abs(signal)) >= 1:
                    signal = signal * MAX_AMP / np.max(np.abs(signal))

            return torch.from_numpy(signal)

        for i, spk_file in enumerate(spk_files):
            # select random offset
            length = torchaudio.info(spk_file).num_frames
            start = 0
            stop = length
            if length > minlen:  # take a random window
                # start = np.random.randint(0, length - minlen)
                start = rng.integers(0, length - minlen)
                stop = start + minlen

            tmp, fs_read = torchaudio.load(
                spk_file, frame_offset=start, num_frames=stop - start,
            )
            tmp = tmp[0].numpy()
            tmp = resample_poly(tmp,hparams["sample_rate"],fs_read)
            tmp = normalize(tmp)
            sources.append(tmp)

        sources = torch.stack(sources)
        mixture = torch.sum(sources, 0)

        # check for clipping
        max_amp_insig = mixture.abs().max().item()
        if max_amp_insig > MAX_AMP:
            weight = MAX_AMP / max_amp_insig
        else:
            weight = 1

        sources = weight * sources
        mix_sig = weight * mixture

        yield mix_sig.float()
        for i in range(hparams["num_spks"]):
            yield sources[i].float()

    sb.dataio.dataset.add_dynamic_item(normal_data, audio_pipeline_mix)    
    sb.dataio.dataset.set_output_keys(
        normal_data, ["id", "mix_sig", "s1_sig", "s2_sig"]
    )

    sb.dataio.dataset.add_dynamic_item(strong_data, audio_pipeline_mix)
    sb.dataio.dataset.set_output_keys(
        strong_data, ["id", "mix_sig", "s1_sig", "s2_sig"]
    )

    return normal_data, strong_data


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Logger info
    logger = logging.getLogger(__name__)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Check if wsj0_tr is set with dynamic mixing
    if hparams["dynamic_mixing"] and not os.path.exists(
        hparams["base_folder_dm"]
    ):
        print(
            "Please, specify a valid base_folder_dm folder when using dynamic mixing"
        )
        sys.exit(1)

    # Data preparation
    # from recipes.LibriMix.prepare_data import prepare_librimix

    # run_on_main(
    #     prepare_librimix,
    #     kwargs={
    #         "datapath": hparams["data_folder"],
    #         "savepath": hparams["save_folder"],
    #         "n_spks": hparams["num_spks"],
    #         "skip_prep": hparams["skip_prep"],
    #         "librimix_addnoise": hparams["use_wham_noise"],
    #         "fs": hparams["sample_rate"],
    #     },
    # )
    
    from RAVDESS2Mix_BSS_prep import genEmoWise
    
    if hparams["speaker_list"]=="partial":
        speaker_list = [*range(1,25,3)]
    elif hparams["speaker_list"]=="full":
        speaker_list = [*range(1,25)]
        
        
    
    if not hparams["skip_prep"]:
        run_on_main(
            genEmoWise,
            kwargs={
                "datapath": hparams["data_folder"],
                "savepath": hparams["save_folder"],
                "speaker_list": speaker_list #[*range(1,25,3)] if not specified. Earlier version of hparams file did not specify
            }
        )
    emotion_list = [*range(1,9)]
    emotion_combs = [*itertools.product(emotion_list,emotion_list)]
    hparams["normal_emotions_list"] = [getEmotion(x[0])+"_"+getEmotion(x[1]) for x in emotion_combs]

    emotion_list = [*range(2,9)]
    emotion_combs = [*itertools.product(emotion_list,emotion_list)]
    hparams["strong_emotions_list"] = [getEmotion(x[0])+"_"+getEmotion(x[1]) for x in emotion_combs]
    
    normal_data, strong_data = dataio_prep(hparams)

    # Load pretrained model if pretrained_separator is present in the yaml
    if "pretrained_separator" in hparams:
        run_on_main(hparams["pretrained_separator"].collect_files)
        hparams["pretrained_separator"].load_collected()

    # Brain class initialization
    separator = Separation(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # re-initialize the parameters if we don't use a pretrained model
    if "pretrained_separator" not in hparams:
        for module in separator.modules.values():
            separator.reset_layer_recursively(module)

    if not hparams["test_only"]:
        # Training
        separator.fit(
            separator.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["dataloader_opts"],
            valid_loader_kwargs=hparams["dataloader_opts"],
        )

    summary_file = os.path.join(hparams["output_folder"], f'summary_results.csv')
    summary_columns = ["emotion", "sdr", "sdr_i", "si-snr", "si-snr_i"]
    print(summary_file)
    with open(summary_file, "w") as summary_csv:
        writer = csv.DictWriter(summary_csv, fieldnames=summary_columns)
        writer.writeheader()
    
    # Eval
    
    for i in range(len(hparams["normal_emotions_list"])):
        separator.evaluate(normal_data[i], min_key="si-snr")
        separator.save_results(normal_data[i], f'nn_{hparams["normal_emotions_list"][i]}')
    
    for i in range(len(hparams["strong_emotions_list"])):
            separator.evaluate(strong_data[i], min_key="si-snr")
            separator.save_results(strong_data[i], f'ss_{hparams["strong_emotions_list"][i]}')
    print("done")

