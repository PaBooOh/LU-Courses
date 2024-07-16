import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

FRAME_SIZE = 512
N_BANDS = 8
y, sr = librosa.load(path='piano.wav', sr=40000)

S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
# S.shape
fig, ax = plt.subplots()
S_dB = librosa.power_to_db(S, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=sr,
                         fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency spectrogram')
m = np.argmax(S, axis=0)
C = []

for i in range(1, m.shape[0]):
    if m[i-1] < m[i]:
        C.append('Up')
    elif m[i-1] > m[i]:
        C.append('Down')
    else:
        C.append('Repeat')

print(m)

