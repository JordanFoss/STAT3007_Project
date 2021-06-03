import glob
import numpy as np
import soundfile as sf
import os
import argparse
import librosa
import configparser as CP
from audiolib import audioread, audiowrite, snr_mixer

def main(cfg):

    statement_to_use = '02'
    
    # the output directory
    output_dir = str(cfg['output_dir'])
    # clean speech base directory
    clean_dir = str(cfg['speech_dir'])
    # noise train directory
    noise_dir = os.path.join(os.path.dirname(__file__), 'noise_train')

    for i in ['Female', 'Male']:
        path = os.path.join(clean_dir, i)
        output_dir_1 = os.path.join(output_dir, i)
        for actor_name in os.listdir(path):
            clean_to_use = []
            actor_dir = os.path.join(path, actor_name)
            output_dir_2 = os.path.join(output_dir_1, actor_name)
            if not os.path.exists(output_dir_2):
                os.mkdir(output_dir_2)
            for clean_file in glob.glob(os.path.join(actor_dir, '*.wav')):
                basename = os.path.basename(clean_file)
                series = basename.split(".")[0].split("-")
                if series[2] == statement_to_use:
                    print(basename)
                    clean_to_use.append(clean_file)
            print(len(clean_to_use))
            for f_clean in clean_to_use:
                clean_samples, clean_sampling_rate = librosa.load(f_clean, sr = None)
                for f_noise in glob.glob(os.path.join(noise_dir, '*.wav')):
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
                    clean_snr, noise_snr, noisy_snr = snr_mixer(clean=clean_samples, noise=noisy_samples, snr=30)
                    noise_name = os.path.basename(f_noise).split(".")[0]
                    clean_name = os.path.basename(f_clean).split(".")[0]
                    output_filename = clean_name + '_' + noise_name + '.wav'
                    output_filename = os.path.join(output_dir_2, output_filename)
                    audiowrite(noisy_snr, 48000, output_filename, norm=False)
            print(actor_name + ' done')


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