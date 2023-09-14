## Must always be a mixture of different statements
## we will use all speaker combinations. Left column will be statement1, right column will be statement 2. 
## we will use all combinations of repetitions, i.e. 01-01, 02-02,01-02,02-01. = 4
## we will use all combinations of all emotions 8 * 8 = 64

'''
This code is responsible for creating mixes of the RAVDESS for blind source separation (BSS)

The RAVDESS dataset can be obtained from
https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

We use the same subset of the RAVDESS dataset as the authors of RAVDESS2Mix


THe 
[mode]-[speech/song]-[EMOTION]-[INTENSITY]-[STATEMENT]-[REP]-[ID].wav

'''
import os
import itertools
import random
import csv

def get_data_list(data_folder):
    files = []
    for i in range(1,25):
        subfolder = f'Actor_{i}' if i>9 else f'Actor_0{i}'
        wav_list = os.listdir(f'{data_folder}/{subfolder}')
        for item in wav_list:
            files.append(subfolder+'/'+item)
    return files

def getEmotion(code):
    '''
    dictionary function to convert between filename identifier code and the emotion.
    '''
    code_dict={
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised',
    }
    if type(code)==str:
        return code_dict[code]
    elif type(code)==int:
        return [*code_dict.values()][code-1]
    else:
        raise Exception("type must be int or str")
        
def getIntensity(code):
    '''
    dictionary function to convert between filename identifier code and emotional intensity.
    '''
    code_dict={
        '01': 'normal',
        '02': 'strong'
    }
    if type(code)==str:
        return code_dict[code]
    elif type(code)==int:
        return [*code_dict.values()][code-1]
    else:
        raise Exception("type must be int or str")
        
def pad_code(code: int):
    '''
    This function converts and integer to a 2 character code.
    '''
    if code>9:
        return f'{code}'
    else:
        return f'0{code}'

def speaker_mixer(emotion_combs, 
                  speaker_combs, 
                  ity = [1,1], 
                  repetitions=[*itertools.product([*range(1,3)],[*range(1,3)])], 
                  seed=1234):
    '''
    emotion_combs should be a list, which each item being a list of length 2
    speaker_combs should be a list, which each item being a list of length 2
    ity should be a list of lenght 2. Options are [1,1] [1,2] [2,1] [2,2]
    repetitions selects which repetition of the sentence should be used
    seed for the random generator instantiated within the function
    '''
    
    #Setting up variables for loop
    lft = [] #statement 1
    rgt = [] #statement 2
    if seed is not None:
        rgen = random.Random(seed)

    for emo in emotion_combs:
        for spk in speaker_combs:
            selected_repetitions = repetitions if seed==None else [repetitions[rgen.randint(0,len(repetitions)-1)]]
            for rep in selected_repetitions:
                lft.append(f'Actor_{pad_code(spk[0])}/' +
                           "-".join(map(str, [pad_code(i) for i in [3,1,emo[0],ity[0],1,rep[0],spk[0]]])) +
                           ".wav"
                          )
                rgt.append(f'Actor_{pad_code(spk[1])}/' +
                           "-".join(map(str, [pad_code(i) for i in [3,1,emo[1],ity[1],2,rep[1],spk[1]]])) +
                           ".wav"
                          )    
    return lft, rgt

def genEmoWise(datapath,
               savepath, 
               speaker_list = [*range(1,25)], 
               intensities = [[1,1],[2,2]]
              ):
    '''
    only normal_normal and strong_strong emotional intensities
    Mixted emotional intensities must be handled by a different function
    '''
    if not savepath[-1]=="/": 
        savepath=savepath+"/"
    
    if not datapath[-1]=="/": 
        datapath=datapath+"/"
    
    speaker_combs = [*itertools.permutations(speaker_list,2)] #Permutations gives us the full set

 

    csv_columns = ['ID','s1_wav','s2_wav']
    
    for ity in intensities:
        ity_comb_name = getIntensity(ity[0])+'_'+getIntensity(ity[1])
        
        if not os.path.isdir(savepath + ity_comb_name): os.makedirs(savepath + ity_comb_name)
        
        emotion_list_1 = [*range(1,9)] if ity[0]==1 else [*range(2,9)]
        emotion_list_2 = [*range(1,9)] if ity[1]==1 else [*range(2,9)]
        
        emotion_combs = [*itertools.product(emotion_list_1,emotion_list_2)] 
        
        for emo in emotion_combs:
            emo_comb_name = getEmotion(emo[0])+'_'+getEmotion(emo[1])
    
            left, right = speaker_mixer([emo], speaker_combs, ity)
        
            with open(savepath + ity_comb_name + "/"+ emo_comb_name + ".csv", "w") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
                for i, (left, right) in enumerate(
                    zip(left, right)
                    ):
                    row = {
                        'ID': i,
                        's1_wav': datapath+left,
                        's2_wav': datapath+right
                    }
                    writer.writerow(row)
                    
if __name__ == '__main__':
    genEmoWise("/home/jiaqi.yip/code/RAVDESS2Mix/RAVDESS/","RAVDESS2Mix_csv/sep_full_spks")
    genEmoWise("/home/jiaqi.yip/code/RAVDESS2Mix/RAVDESS/","RAVDESS2Mix_csv/sep_third_spks",speaker_list = [*range(1,25,3)])