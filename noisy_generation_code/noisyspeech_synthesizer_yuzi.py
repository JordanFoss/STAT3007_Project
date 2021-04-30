"""
@author: chkarada
"""
import glob
import numpy as np
import soundfile as sf
import os
import argparse
import configparser as CP
from audiolib import audioread, audiowrite, snr_mixer

def main(cfg):
    # This is not used because we only want one SNR value, not a list of SNR values
    snr_lower = float(cfg["snr_lower"])
    snr_upper = float(cfg["snr_upper"])
    total_snrlevels = float(cfg["total_snrlevels"])
    
    # the output directory
    output_dir = str(cfg['output_dir'])
    # clean speech base directory
    clean_dir = str(cfg['speech_dir'])
    # female actor speech directory
    female_dir = os.path.join(clean_dir, 'Female')
    # male actor speech directory
    male_dir = os.path.join(clean_dir, 'Male')
    
    # noise train directory
    noise_dir = os.path.join(os.path.dirname(__file__), 'noise_train')

    # make output directory for each actor
    for dir in [female_dir, male_dir]:
        gender = os.path.basename(dir)
        actor_names = os.listdir(dir)
        for actor in actor_names:
            actor_output_dir = os.path.join(output_dir, gender)
            actor_output_dir = os.path.join(actor_output_dir, actor)

            # print(actor_output_dir)
            if not os.path.exists(actor_output_dir):
                os.mkdir(actor_output_dir)
            
            fs = float(cfg["sampling_rate"])
            audioformat = cfg["audioformat"]
            silence_length = float(cfg["silence_length"])

            actor_dir = os.path.join(dir, actor)
            # SNR = np.linspace(snr_lower, snr_upper, int(total_snrlevels))
            cleanfilenames = glob.glob(os.path.join(actor_dir, audioformat))

            if cfg["noise_types_excluded"]=='None':
                noisefilenames = glob.glob(os.path.join(noise_dir, audioformat))
            else:
                filestoexclude = cfg["noise_types_excluded"].split(',')
                noisefilenames = glob.glob(os.path.join(noise_dir, audioformat))
                for i in range(len(filestoexclude)):
                    noisefilenames = [fn for fn in noisefilenames if not os.path.basename(fn).startswith(filestoexclude[i])]

            filecounter = 0
            num_samples = 0

            for idx_s in range(len(cleanfilenames)):
                clean, fs = audioread(cleanfilenames[idx_s])
                
                for idx_n in range(len(noisefilenames)):
                    noise, fs = audioread(noisefilenames[idx_n])

                    if len(noise)>=len(clean):
                        noise = noise[0:len(clean)]
                    
                    else:
                    
                        while len(noise)<=len(clean):
                            newnoise, fs = audioread(noisefilenames[idx_n])
                            noiseconcat = np.append(noise, np.zeros(int(fs*silence_length)))
                            noise = np.append(noiseconcat, newnoise)
                    noise = noise[0:len(clean)]
                    filecounter = filecounter + 1

                    clean_snr, noise_snr, noisy_snr = snr_mixer(clean=clean, noise=noise, snr=0)
                    cleanfilename = os.path.basename(cleanfilenames[idx_s]).split('.')[0]

                    noisefilename = os.path.basename(noisefilenames[idx_n]).split('.')[0]

                    combinedfilename = cleanfilename + '_' + noisefilename + '.wav'

                    noisypath = os.path.join(actor_output_dir, combinedfilename)
                    # here 48000 is the frequency of the clean speech
                    audiowrite(noisy_snr, 48000, noisypath, norm=False)
            
            

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
    