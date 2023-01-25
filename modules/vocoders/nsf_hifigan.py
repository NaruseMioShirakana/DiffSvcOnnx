import os

import torch

from modules.nsf_hifigan.models import load_model
from modules.nsf_hifigan.nvSTFT import load_wav_to_torch, STFT
from utils.hparams import hparams

nsf_hifigan = None


def register_vocoder(cls):
    global nsf_hifigan
    nsf_hifigan = cls
    return cls


@register_vocoder
class NsfHifiGAN():
    def __init__(self, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        model_path = hparams['vocoder_ckpt']
        if os.path.exists(model_path):
            print('| Load HifiGAN: ', model_path)
            self.model, self.h = load_model(model_path, device=self.device)
        else:
            print('Error: HifiGAN model file is not found!')

    def spec2wav(self, mel, **kwargs):
        with torch.no_grad():
            c = mel.to(self.device)
            f0 = kwargs.get('f0')
            f0 = f0.to(self.device)
            if False:
                print(f0.shape, c.shape)
                torch.onnx.export(
                    self.model,
                    (c, f0),
                    "nsf_hifigan.onnx",
                    input_names=["c", "f0"],
                    output_names=["audio"],
                    dynamic_axes={
                        "c": [2],
                        "f0": [1]
                    },
                    opset_version=16
                )
            y = self.model(c, f0).view(-1)
            print(y.shape)
        wav_out = y.cpu().numpy()
        return wav_out

    @staticmethod
    def wav2spec(inp_path, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sampling_rate = hparams['audio_sample_rate']
        num_mels = hparams['audio_num_mel_bins']
        n_fft = hparams['fft_size']
        win_size = hparams['win_size']
        hop_size = hparams['hop_size']
        fmin = hparams['fmin']
        fmax = hparams['fmax']
        stft = STFT(sampling_rate, num_mels, n_fft, win_size, hop_size, fmin, fmax)
        with torch.no_grad():
            wav_torch, _ = load_wav_to_torch(inp_path, target_sr=stft.target_sr)
            mel_torch = stft.get_mel(wav_torch.unsqueeze(0).to(device)).squeeze(0).T
            # log mel to log10 mel
            mel_torch = 0.434294 * mel_torch
            return wav_torch.cpu().numpy(), mel_torch.cpu().numpy()
