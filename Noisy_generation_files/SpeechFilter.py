import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import freqz
import scipy.signal as sg

speech_path = '/Volumes/YuziSSD1/STAT3007/project/Noisy_Speech_Actors_01-24/Female/Actor_02/01-01-01-01-02_AirConditioner_1.wav'
dest_path = '/Volumes/YuziSSD1/STAT3007/project/MS-SNSD/filtered.wav'


x, fr = sf.read(speech_path)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
t = np.linspace(0., len(x) / fr, len(x))
ax.plot(t, x, lw=1)

plt.show()


b, a = sg.butter(4, 1000. / (fr / 2.), 'high')
x_fil = sg.filtfilt(b, a, x)
sf.write(dest_path, x_fil, fr)
fig, ax = plt.subplots(1, 1, figsize=(6, 3))
ax.plot(t, x, lw=1)
ax.plot(t, x_fil, lw=1)
plt.show()