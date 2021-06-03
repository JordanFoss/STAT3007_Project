# This file created on 2 June 2021

import glob
import numpy as np
from numpy.core.fromnumeric import repeat
import soundfile as sf
import os
import argparse
import librosa
import configparser as CP
from audiolib import audioread, audiowrite, snr_mixer


def main(cfg):

    # noise train directory
    noise_dir = os.path.join(os.path.dirname(__file__), 'noise_train')
    # the output directory
    output_dir = str(cfg['output_dir'])
    # clean speech base directory
    clean_dir = str(cfg['speech_dir'])

    actors = np.asarray(['01', '02', '03', '04', 
          '05', '06', '07', '08',
          '09', '10', '11', '12',
          '13', '14', '15', '16',
          '17', '18', '19', '20',
          '21', '22', '23', '24'])

    emotion_to_use = ['02','03','04','05','08']
    intensity_to_use = ['02']
    repeatition_to_use = ['02']

    # for actor in actors:
    #     # print(actor)
    #     clean_list = []
    #     if int(actor) % 2 == 0:
    #         clean_dir2 = os.path.join(clean_dir, 'Female')
    #     else:
    #         clean_dir2 = os.path.join(clean_dir, 'Male')

    #     clean_dir3 = os.path.join(clean_dir2, 'Actor_'+actor)
    #     for clean_file in glob.glob(os.path.join(clean_dir3, '*.wav')):
    #         basename = os.path.basename(clean_file)
    #         series = basename.split(".")[0].split("-")
    #         if series[0] in emotion_to_use:
    #             if series[1] in intensity_to_use:
    #                 if series[3] in repeatition_to_use:
    #                     clean_list.append(clean_file) 
    #     print('============')
    #     print(actor)
    #     print(len(clean_list))
    #     clean_list = np.asarray(clean_list)
    #     selected_clean_samples = clean_list[np.random.choice(len(clean_list), size=5, replace=False)]
    #     print(len(selected_clean_samples))
    count = 0
    for f_noise in glob.glob(os.path.join(noise_dir, '*.wav')):
        selected_actors = actors[np.random.choice(len(actors), size=3, replace=False)]
        for actor in selected_actors:
            clean_list = []
            if int(actor) % 2 == 0:
                clean_dir2 = os.path.join(clean_dir, 'Female')
            else:
                clean_dir2 = os.path.join(clean_dir, 'Male')

            clean_dir3 = os.path.join(clean_dir2, 'Actor_'+actor)
            for clean_file in glob.glob(os.path.join(clean_dir3, '*.wav')):
                basename = os.path.basename(clean_file)
                series = basename.split(".")[0].split("-")
                if series[0] in emotion_to_use:
                    if series[1] in intensity_to_use:
                        if series[3] in repeatition_to_use:
                            clean_list.append(clean_file) 
            print(len(clean_list))
            clean_list = np.asarray(clean_list)
            selected_clean_samples = clean_list[np.random.choice(len(clean_list), size=5, replace=False)]
            for f_clean in selected_clean_samples:
                clean_samples, clean_sampling_rate = librosa.load(f_clean, sr = None)
                noisy_samples, noisy_sampling_rate = librosa.load(f_noise, sr = clean_sampling_rate)
                # cut the noise samples, will cut further
                noisy_samples = noisy_samples[:noisy_sampling_rate * ((len(clean_samples)//clean_sampling_rate) + 1)]
                if len(noisy_samples)>=len(clean_samples):
                                # print("Yes!")
                    noisy_samples = noisy_samples[0:len(clean_samples)]
                else:
                    while len(noisy_samples)<len(clean_samples):
                        newnoise, fs = librosa.load(f_noise)
                        noiseconcat = np.append(noise, np.zeros(int(fs*0)))
                        noise = np.append(noiseconcat, newnoise)

                noisy_samples = noisy_samples[0:len(clean_samples)]
                clean_snr, noise_snr, noisy_snr = snr_mixer(clean=clean_samples, noise=noisy_samples, snr=40)
                noise_name = os.path.basename(f_noise).split(".")[0]
                clean_name = os.path.basename(f_clean).split(".")[0]
                output_filename = clean_name + '_' + noise_name + '.wav'
                output_filename = os.path.join(output_dir, output_filename)
                audiowrite(noisy_snr, 48000, output_filename, norm=False)
        count += 1
        print(count)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    # Configurations: read noisyspeech_synthesizer.cfg
    parser.add_argument("--cfg", default = "noisyspeech_synthesizer.cfg", help = "Read noisyspeech_synthesizer.cfg for all the details")
    parser.add_argument("--cfg_str", type=str, default = "noisy_speech" )
    args = parser.parse_args()

    
    cfgpath = os.path.join(os.path.dirname(__file__), args.cfg)
    assert os.path.exists(cfgpath), f"No configuration file as [{cfgpath}]"
    cfg = CP.ConfigParser()
    cfg._interpolation = CP.ExtendedInterpolation()
    cfg.read(cfgpath)
    
    main(cfg._sections[args.cfg_str])