# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 16:35:48 2023

@author: Maxence
"""
import pyroomacoustics as pra
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import IPython
import soundfile as sf
from tqdm import tqdm

from nara_wpe.wpe import wpe
from nara_wpe.wpe import get_power
from nara_wpe.utils import stft, istft, get_stft_center_frequencies
from nara_wpe import project_root

# The desired reverberation time and dimensions of the room
rt60_tgt = 2  # seconds
room_dim = [5, 10, 7]  # meters


filename = "test.wav"
output_filename = "testrev.wav"


fs,audio = wavfile.read(filename) # could use any type of music file (with sf.read)

audio = audio[:,0] # stereo to mono

# We invert Sabine's formula to obtain the parameters for the ISM simulator
e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

# Create the room
room = pra.ShoeBox(
    room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order)

# place the source in the room
room.add_source([2.5, 3.73, 1.76], signal=audio, delay=0.5)

# define the locations of the microphones
mic_locs = np.c_[
    [1,1,1],[2,5,3]  # mic 1  # mic 2
]

# finally place the array in the room
room.add_microphone_array(mic_locs)

# Run the simulation
room.simulate()

room.mic_array.to_wav(output_filename,norm=True,bitdepth=np.int16)

# measure the reverberation time
rt60 = room.measure_rt60()
print("The desired RT60 was {}".format(rt60_tgt))
print("The measured RT60 is {}".format(rt60[1, 0]))

# Create a plot
plt.figure()

# plot one of the RIR. both can also be plotted using room.plot_rir()
rir_1_0 = room.rir[1][0]
plt.subplot(2, 1, 1)
plt.plot(np.arange(len(rir_1_0)) / room.fs, rir_1_0)
plt.title("The RIR from source 0 to mic 1")
plt.xlabel("Time [s]")

# get the mic signal
y = room.mic_array.signals[1, :]

# plot signal at microphone 1
plt.subplot(2, 1, 2)
plt.plot(y)
plt.title("Microphone 1 signal")
plt.xlabel("Time [s]")

plt.tight_layout()
plt.show()

