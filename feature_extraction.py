import librosa
import os
import numpy as np
import matplotlib.pyplot as plt

path = '/home/aliaamahgoub/acoustic_scenes/TAU-urban-acoustic-scenes-2019-development/audio/'

centroid = []
bandwidth = []
for f in os.listdir(path)[:20]:
    print(f)
    x, fs = librosa.load(path + f)
    centroid.append(np.mean(librosa.feature.spectral_centroid(x)))
    bandwidth.append(np.mean(librosa.feature.spectral_bandwidth(x)))

plt.scatter(centroid, bandwidth)
plt.show()
