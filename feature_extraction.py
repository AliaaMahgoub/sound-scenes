import librosa
import os
import numpy as np
import matplotlib.pyplot as plt

path = '/home/aliaamahgoub/acoustic-scenes/TAU-urban-acoustic-scenes-2019-development/audio/'

centroid_airport = []
centroid_bus = []
centroid_metro = []
centroid_shopping_mall = []
centroid_street_pedestrian = []

bandwidth_airport = []
bandwidth_bus = []
bandwidth_metro = []
bandwidth_shopping_mall = []
bandwidth_street_pedestrian = []
# TODO: extract 2 time-based features
# TODO: extract all 4 features for the 10 categories
for f in os.listdir(path):
    print(f)
    x, fs = librosa.load(path + f)
    if 'airport' in f:
        centroid_airport.append(np.mean(librosa.feature.spectral_centroid(x)))
        bandwidth_airport.append(np.mean(librosa.feature.spectral_bandwidth(x)))
    if 'bus' in f:
        centroid_bus.append(np.mean(librosa.feature.spectral_centroid(x)))
        bandwidth_bus.append(np.mean(librosa.feature.spectral_bandwidth(x)))
    if 'metro' in f:
        centroid_metro.append(np.mean(librosa.feature.spectral_centroid(x)))
        bandwidth_metro.append(np.mean(librosa.feature.spectral_bandwidth(x)))
    if 'shopping_mall' in f:
        centroid_shopping_mall.append(np.mean(librosa.feature.spectral_centroid(x)))
        bandwidth_shopping_mall.append(np.mean(librosa.feature.spectral_bandwidth(x)))
    if 'street_pedestrian' in f:
        centroid_street_pedestrian.append(np.mean(librosa.feature.spectral_centroid(x)))
        bandwidth_street_pedestrian.append(np.mean(librosa.feature.spectral_bandwidth(x)))

np.save('centroid_airport', np.array(centroid_airport))
np.save('bandwidth_airport', np.array(bandwidth_airport))
np.save('centroid_bus',np.array(centroid_bus))
np.save('bandwidth_bus', np.array(bandwidth_bus))
np.save('centroid_metro',np.array(centroid_metro))
np.save('bandwidth_metro', np.array(bandwidth_metro))
np.save('centroid_shopping_mall',np.array(centroid_shopping_mall))
np.save('bandwidth_shopping_mall',np.array(bandwidth_shopping_mall))
np.save('centroid_street_pedestrian', np.array(centroid_street_pedestrian))
np.save('bandwidth_street_pedestrian',np.array(bandwidth_street_pedestrian))

plt.scatter(centroid_airport, bandwidth_airport, color = 'red', label= 'airport')
plt.scatter(centroid_bus, bandwidth_bus, color = 'blue', label='bus')
plt.scatter(centroid_metro, bandwidth_metro, color = 'green', label='metro')
plt.scatter(centroid_shopping_mall, bandwidth_shopping_mall, color = 'black', label='shopping mall')
plt.scatter(centroid_street_pedestrian, bandwidth_street_pedestrian, color = 'orange', label='pedestrian street')
plt.scatter
plt.legend()
plt.show()
