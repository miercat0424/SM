import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import skimage.io
import numpy as np
from glob import glob

from torchaudio.transforms import MelSpectrogram

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def spectrogram_image(y, sr, out, hop_length, n_mels):
    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                            n_fft=hop_length*2, hop_length=hop_length)
    mels = np.log(mels + 1e-9) # add small number to avoid log(0)

    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0) # put low frequencies at the bottom in image
    img = 255-img # invert. make black==more energy

    # save as PNG
    skimage.io.imsave(out, img)

n_fft = 22050
win_len = None
hop_len = 512
n_mels = 128
sample_rate = 22050
wavs = glob("WAV_Files/*.wav")
paths = [path.replace("WAV_Files/","") for path in wavs]
count = 0

for path in paths :
    
    waveform, sample_rate = librosa.load(path, sr=sample_rate)
    waveform = torch.Tensor(waveform)

    torchaudio_melspec = MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_len,
        hop_length=hop_len,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm='slaney',
        onesided=True,
        n_mels=n_mels,
    )(waveform)

    # mse = ((torchaudio_melspec - librosa_melspec) ** 2).mean()

    # print(f'MSE:\t{mse}')

    # fig, axs = plt.subplots(1, 2, figsize=(20, 5))
    # fig.suptitle('Mel Spectrogram')

    # axs[0].set_title('torchaudio')
    # axs[0].set_ylabel('Log')
    # axs[0].set_xlabel('time')
    # # axs[0].imshow(librosa.amplitude_to_db(torchaudio_melspec), aspect='auto')
    # Data = librosa.amplitude_to_db(torchaudio_melspec)
    # img1 = librosa.display.specshow(Data,y_axis="log",x_axis="time",sr=sample_rate,ax=axs[0])

    # fig.savefig("spec.png")
    out = f"MelGray_Files/spectro{count}.png"

    start_sample = 0 # starting at beginning
    length_samples = 390*hop_len
    window = waveform[start_sample:start_sample+length_samples]

    spectrogram_image(window.numpy(), sr=sample_rate, out=out, hop_length=hop_len, n_mels=n_mels)
    count += 1 

