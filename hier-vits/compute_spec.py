# from scipy.io.wavfile import read
import librosa
import torch
import numpy as np
import os
from multiprocessing import Pool
from tqdm import tqdm

# Change here
base= '/nfs-data-2/younghun/speech/genesislab/announcer/studio_202201/preprocessed/sr_22050_enhanced/aligned_data'

hann_window = {}
def load_wav_to_torch(full_path):
    data, sampling_rate = librosa.load(full_path)
    # data, sampling_rate = librosa.load(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate

def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec

def get_audio(filename):
    max_wave_length = 32768.0
    filter_length = 1024
    hop_length = 256
    win_length = 1024
    save_dir = "/data/jaeyoung/speech_synthesis/VITS/studio_spec"
    audio, sampling_rate = load_wav_to_torch(filename)
    audio_norm = audio / max_wave_length
    audio_norm = audio_norm.unsqueeze(0)
    _file = filename.split('/')[-1].split(".")[0]
    spec_filename = os.path.join(save_dir, f'{_file}.spec.pt')
    spec = spectrogram_torch(audio_norm, filter_length,
        sampling_rate, hop_length, win_length,
        center=False)
    spec = torch.squeeze(spec, 0)
    torch.save(spec, spec_filename)

if __name__=="__main__":
    waves = []
    speakers = ['정윤성', '이승은', '임지윤']
    for s in speakers:
        s_audio = os.path.join(base, s)
        wav_path = os.listdir(s_audio)
        for wav in wav_path:
            if wav.endswith(".flac"):
                waves.append(os.path.join(s_audio, wav))

    with Pool(16) as p:
        print(list((tqdm(p.imap(get_audio, waves), total=len(waves)))))