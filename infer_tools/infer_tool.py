import json
import os
import time
from io import BytesIO
from pathlib import Path

import librosa
import numpy as np
import soundfile
import torch

import utils
from infer_tools.f0_static import compare_pitch, static_f0_time
from modules.diff.diffusion import GaussianDiffusion
from modules.diff.net import DiffNet
from modules.vocoders.nsf_hifigan import NsfHifiGAN
from preprocessing.hubertinfer import HubertEncoder
from preprocessing.process_pipeline import File2Batch, get_pitch_parselmouth
from utils.hparams import hparams, set_hparams
from utils.pitch_utils import denorm_f0, norm_interp_f0


def timeit(func):
    def run(*args, **kwargs):
        t = time.time()
        res = func(*args, **kwargs)
        print('executing \'%s\' costed %.3fs' % (func.__name__, time.time() - t))
        return res

    return run


def format_wav(audio_path):
    if Path(audio_path).suffix == '.wav':
        return
    raw_audio, raw_sample_rate = librosa.load(audio_path, mono=True, sr=None)
    soundfile.write(Path(audio_path).with_suffix(".wav"), raw_audio, raw_sample_rate)


def fill_a_to_b(a, b):
    if len(a) < len(b):
        for _ in range(0, len(b) - len(a)):
            a.append(a[0])


def get_end_file(dir_path, end):
    file_lists = []
    for root, dirs, files in os.walk(dir_path):
        files = [f for f in files if f[0] != '.']
        dirs[:] = [d for d in dirs if d[0] != '.']
        for f_file in files:
            if f_file.endswith(end):
                file_lists.append(os.path.join(root, f_file).replace("\\", "/"))
    return file_lists


def mkdir(paths: list):
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)


class Svc:
    def __init__(self, project_name, config_name, hubert_gpu, model_path, onnx=False):
        self.project_name = project_name
        self.DIFF_DECODERS = {
            'wavenet': lambda hp: DiffNet(hp['audio_num_mel_bins']),
        }

        self.model_path = model_path
        self.dev = torch.device("cuda")

        self._ = set_hparams(config=config_name, exp_name=self.project_name, infer=True,
                             reset=True, hparams_str='', print_hparams=False)

        self.mel_bins = hparams['audio_num_mel_bins']
        hparams['hubert_gpu'] = hubert_gpu
        self.hubert = HubertEncoder(hparams['hubert_path'], onnx=onnx)
        self.model = GaussianDiffusion(
            phone_encoder=self.hubert,
            out_dims=self.mel_bins, denoise_fn=self.DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
            timesteps=hparams['timesteps'],
            K_step=hparams['K_step'],
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
        )
        utils.load_ckpt(self.model, self.model_path, 'model', force=True, strict=True)
        self.model.cuda()
        self.vocoder = NsfHifiGAN()

    def infer(self, in_path, key, acc, spk_id=0, use_crepe=True):
        batch = self.pre(in_path, acc, spk_id, use_crepe)
        batch['f0'] = batch['f0'] + (key / 12)
        batch['f0'][batch['f0'] > np.log2(hparams['f0_max'])] = 0

        spk_embed = batch.get('spk_embed') if not hparams['use_spk_id'] else batch.get('spk_ids')
        if spk_embed is None:
            spk_embed = torch.LongTensor([0]).cuda()
        speedup = torch.LongTensor([acc]).cuda()
        initial_noise = torch.randn((1, 1, self.model.mel_bins, batch['f0'].shape[1])).cuda()
        ONNX = True
        if ONNX:
            torch.onnx.export(
                self.model,
                (
                    batch['hubert'].cuda(),
                    batch['mel2ph'].cuda(),
                    spk_embed.cuda(),
                    batch['f0'].cuda(),
                    initial_noise,
                    speedup
                ),
                "ShirohaSvc_DiffSvc.onnx",
                input_names=["hubert", "mel2ph", "spk_embed", "f0", "initial_noise", "speedup"],
                output_names=["mel_pred", "f0_pred"],
                dynamic_axes={
                    "hubert": [1],
                    "f0": [1],
                    "mel2ph": [1],
                    "initial_noise": [3]
                },
                opset_version=16
            )
        mel_pred, f0_pred = self.model(
            batch['hubert'].cuda(), batch['mel2ph'].cuda(), spk_embed.cuda(), batch['f0'].cuda(), initial_noise, speedup
        )
        wav_pred = self.vocoder.spec2wav(mel_pred, f0=f0_pred)
        return wav_pred

    def pre(self, wav_fn, accelerate, spk_id=0, use_crepe=True):
        if isinstance(wav_fn, BytesIO):
            item_name = self.project_name
        else:
            song_info = wav_fn.split('/')
            item_name = song_info[-1].split('.')[-2]
        temp_dict = {'wav_fn': wav_fn, 'spk_id': spk_id, 'id': 0}

        temp_dict = File2Batch.temporary_dict2processed_input(item_name, temp_dict, self.hubert, infer=True,
                                                              use_crepe=use_crepe)
        hparams['pndm_speedup'] = accelerate
        batch = File2Batch.processed_input2batch([getitem(temp_dict)])
        return batch


def getitem(item):
    max_frames = hparams['max_frames']
    spec = torch.Tensor(item['mel'])[:max_frames]
    mel2ph = torch.LongTensor(item['mel2ph'])[:max_frames] if 'mel2ph' in item else None
    f0, uv = norm_interp_f0(item["f0"][:max_frames], hparams)
    hubert = torch.Tensor(item['hubert'][:hparams['max_input_tokens']])
    pitch = torch.LongTensor(item.get("pitch"))[:max_frames]
    sample = {
        "id": item['id'],
        "spk_id": item['spk_id'],
        "item_name": item['item_name'],
        "hubert": hubert,
        "mel": spec,
        "pitch": pitch,
        "f0": f0,
        "uv": uv,
        "mel2ph": mel2ph,
        "mel_nonpadding": spec.abs().sum(-1) > 0,
    }
    if hparams['use_energy_embed']:
        sample['energy'] = item['energy']
    return sample
