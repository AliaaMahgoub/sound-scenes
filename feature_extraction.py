import librosa
import os
import numpy as np
import matplotlib.pyplot as plt

path = '/home/aliaamahgoub/acoustic-scenes/TAU-urban-acoustic-scenes-2019-development/audio/'

centroid_airport = []
centroid_shopping_mall = []
centroid_metro_station = []
centroid_street_pedestrian = []
centroid_public_square = []
centroid_street_traffic = []
centroid_tram =[]
centroid_bus = []
centroid_metro =[]
centroid_park = []

bandwidth_airport = []
bandwidth_shopping_mall = []
bandwidth_metro_station = []
bandwidth_street_pedestrian = []
bandwidth_public_square = []
bandwidth_street_traffic = []
bandwidth_tram = []
bandwidth_bus = []
bandwidth_metro = []
bandwidth_park = []

zc_airport = []
zc_shopping_mall = []
zc_metro_station = []
zc_street_pedestrian = []
zc_public_square = []
zc_street_traffic = []
zc_tram = []
zc_bus = []
zc_metro = []
zc_park = []

rms_airport = []
rms_shopping_mall = []
rms_metro_station = []
rms_street_pedestrian = []
rms_public_square = []
rms_street_traffic = []
rms_tram = []
rms_bus = []
rms_metro = []
rms_park = []

# TODO: extract 2 time-based features
# TODO: extract all 4 features for the 10 categories
for f in os.listdir(path):
    print(f)
    x, fs = librosa.load(path + f)
    if 'airport' in f:
        centroid_airport.append(np.mean(librosa.feature.spectral_centroid(x)))
        bandwidth_airport.append(np.mean(librosa.feature.spectral_bandwidth(x)))
        zc_airport.append(np.mean(librosa.feature.zero_crossing_rate(x)))
        rms_airport.append(np.mean(librosa.feature.rms(x)))
    if 'shopping_mall' in f:
        centroid_shopping_mall.append(np.mean(librosa.feature.spectral_centroid(x)))
        bandwidth_shopping_mall.append(np.mean(librosa.feature.spectral_bandwidth(x)))
        zc_shopping_mall.append(np.mean(librosa.feature.zero_crossing_rate(x)))
        rms_shopping_mall.append(np.mean(librosa.feature.rms(x)))
    if 'metro_station' in f:
        centroid_metro_station.append(np.mean(librosa.feature.spectral_centroid(x)))
        bandwidth_metro_station.append(np.mean(librosa.feature.spectral_bandwidth(x)))
        zc_metro_station.append(np.mean(librosa.feature.zero_crossing_rate(x)))
        rms_metro_station.append(np.mean(librosa.feature.rms(x)))
    if 'street_pedestrian' in f:
        centroid_street_pedestrian.append(np.mean(librosa.feature.spectral_centroid(x)))
        bandwidth_street_pedestrian.append(np.mean(librosa.feature.spectral_bandwidth(x)))
        zc_street_pedestrian.append(np.mean(librosa.feature.zero_crossing_rate(x)))
        rms_street_pedestrian.append(np.mean(librosa.feature.rms(x)))
    if 'public_square' in f:
        centroid_public_square.append(np.mean(librosa.feature.spectral_centroid(x)))
        bandwidth_public_square.append(np.mean(librosa.feature.spectral_bandwidth(x)))
        zc_public_square.append(np.mean(librosa.feature.zero_crossing_rate(x)))
        rms_public_square.append(np.mean(librosa.feature.rms(x)))
    if 'street_traffic' in f:
        centroid_street_traffic.append(np.mean(librosa.feature.spectral_centroid(x)))
        bandwidth_street_traffic.append(np.mean(librosa.feature.spectral_bandwidth(x)))
        zc_street_traffic.append(np.mean(librosa.feature.zero_crossing_rate(x)))
        rms_street_traffic.append(np.mean(librosa.feature.rms(x)))
    if 'tram' in f:
        centroid_tram.append(np.mean(librosa.feature.spectral_centroid(x)))
        bandwidth_tram.append(np.mean(librosa.feature.spectral_bandwidth(x)))
        zc_tram.append(np.mean(librosa.feature.zero_crossing_rate(x)))
        rms_tram.append(np.mean(librosa.feature.rms(x)))
    if 'bus' in f:
        centroid_bus.append(np.mean(librosa.feature.spectral_centroid(x)))
        bandwidth_bus.append(np.mean(librosa.feature.spectral_bandwidth(x)))
        zc_bus.append(np.mean(librosa.feature.zero_crossing_rate(x)))
        rms_bus.append(np.mean(librosa.feature.rms(x)))
    if 'metro-' in f:
        centroid_metro.append(np.mean(librosa.feature.spectral_centroid(x)))
        bandwidth_metro.append(np.mean(librosa.feature.spectral_bandwidth(x)))
        zc_metro.append(np.mean(librosa.feature.zero_crossing_rate(x)))
        rms_metro.append(np.mean(librosa.feature.rms(x)))
    if 'park-' in f:
        centroid_park.append(np.mean(librosa.feature.spectral_centroid(x)))
        bandwidth_park.append(np.mean(librosa.feature.spectral_bandwidth(x)))
        zc_park.append(np.mean(librosa.feature.zero_crossing_rate(x)))
        rms_park.append(np.mean(librosa.feature.rms(x)))

np.save('centroid_airport', np.array(centroid_airport))
np.save('bandwidth_airport', np.array(bandwidth_airport))
np.save('zc_airport', np.array(zc_airport))
np.save('rms_airport', np.array(rms_airport))

np.save('centroid_shopping_mall',np.array(centroid_shopping_mall))
np.save('bandwidth_shopping_mall', np.array(bandwidth_shopping_mall))
np.save('zc_shopping_mall', np.array(zc_shopping_mall))
np.save('rms_shopping_mall', np.array(rms_shopping_mall))

np.save('centroid_metro_station',np.array(centroid_metro_station))
np.save('bandwidth_metro_station',np.array(bandwidth_metro_station))
np.save('zc_metro_station', np.array(zc_metro_station))
np.save('rms_metro_station', np.array(rms_metro_station))

np.save('centroid_street_pedestrian',np.array(centroid_street_pedestrian))
np.save('bandwidth_street_pedestrian',np.array(bandwidth_street_pedestrian))
np.save('zc_street_pedestrian', np.array(zc_street_pedestrian))
np.save('rms_street_pedestrian', np.array(rms_street_pedestrian))

np.save('centroid_public_square', np.array(centroid_public_square))
np.save('bandwidth_public_square',np.array(bandwidth_public_square))
np.save('zc_public_square', np.array(zc_public_square))
np.save('rms_public_square', np.array(rms_public_square))

np.save('centroid_street_traffic', np.array(centroid_street_traffic))
np.save('bandwidth_street_traffic',np.array(bandwidth_street_traffic))
np.save('zc_street_traffic', np.array(zc_street_traffic))
np.save('rms_street_traffic', np.array(rms_street_traffic))

np.save('centroid_tram', np.array(centroid_tram))
np.save('bandwidth_tram',np.array(bandwidth_tram))
np.save('zc_tram', np.array(zc_tram))
np.save('rms_tram', np.array(rms_tram))

np.save('centroid_bus', np.array(centroid_bus))
np.save('bandwidth_bus',np.array(bandwidth_bus))
np.save('zc_bus', np.array(zc_bus))
np.save('rms_bus', np.array(rms_bus))

np.save('centroid_metro', np.array(centroid_metro))
np.save('bandwidth_metro',np.array(bandwidth_metro))
np.save('zc_metro', np.array(zc_metro))
np.save('rms_metro', np.array(rms_metro))

np.save('centroid_park', np.array(centroid_park))
np.save('bandwidth_park',np.array(bandwidth_park))
np.save('zc_park', np.array(zc_park))
np.save('rms_park', np.array(rms_park))

plt.scatter(centroid_airport, bandwidth_airport, color = 'red', label= 'airport')
plt.scatter(centroid_shopping_mall, bandwidth_shopping_mall, color = 'black', label='shopping mall')
plt.scatter(centroid_metro_station, bandwidth_metro_station, color = 'green', label='metro station')
plt.scatter(centroid_street_pedestrian, bandwidth_street_pedestrian, color = 'orange', label='pedestrian street')
plt.scatter(centroid_public_square, bandwidth_public_square, color = 'pink', label='public square')
plt.scatter(centroid_street_traffic, bandwidth_street_traffic, color = 'purple', label='street traffic')
plt.scatter(centroid_tram, bandwidth_tram, color = 'grey', label='tram')
plt.scatter(centroid_bus, bandwidth_bus, color = 'blue', label='bus')
plt.scatter(centroid_metro, bandwidth_metro, color = 'yellow', label='metro')
plt.scatter(centroid_park, bandwidth_park, color = 'c', label='park')

plt.legend()
plt.xlabel('Average Spectral Centroid')
plt.ylabel('Average Spectral Bandwidth')
plt.show()

plt.scatter(zc_airport, rms_airport, color = 'red', label= 'airport')
plt.scatter(zc_shopping_mall, rms_shopping_mall, color = 'black', label='shopping mall')
plt.scatter(zc_metro_station, rms_metro_station, color = 'green', label='metro station')
plt.scatter(zc_street_pedestrian, rms_street_pedestrian, color = 'orange', label='pedestrian street')
plt.scatter(zc_public_square, rms_public_square, color = 'pink', label='public square')
plt.scatter(zc_street_traffic, rms_street_traffic, color = 'purple', label='street traffic')
plt.scatter(zc_tram, rms_tram, color = 'grey', label='tram')
plt.scatter(zc_bus, rms_bus, color = 'blue', label='bus')
plt.scatter(zc_metro, rms_metro, color = 'yellow', label='metro')
plt.scatter(zc_park, rms_park, color = 'c', label='park')

plt.legend()
plt.xlabel('Average Zero Crossing Rate')
plt.ylabel('Average Root Mean Square Energy')
plt.show()
