import io
import os.path
from pathlib import Path

import numpy as np
import soundfile

from infer_tools import infer_tool
from infer_tools.infer_tool import Svc
from infer_tools.trans_key import trans_opencpop
from utils.hparams import hparams


def run_clip(raw_audio_path, svc_model, key, acc, use_crepe, spk_id=0, auto_key=False, use_gt_mel=False,
             add_noise_step=500,
             units_mode=False):
    infer_tool.format_wav(raw_audio_path)
    key = svc_model.evaluate_key(raw_audio_path, key, auto_key)
    _f0_tst, _f0_pred, _audio = svc_model.infer(raw_audio_path, key=key, acc=acc, use_crepe=use_crepe, spk_id=spk_id,
                                                singer=not units_mode, use_gt_mel=use_gt_mel,
                                                add_noise_step=add_noise_step)
    if units_mode:
        out_path = io.BytesIO()
        soundfile.write(out_path, _audio, hparams["audio_sample_rate"], format='wav')
        out_path.seek(0)
        npy_path = Path(raw_audio_path).with_suffix(".npy")
        np.save(str(npy_path), svc_model.hubert.encode(out_path))
    else:
        out_path = f'./singer_data/{Path(raw_audio_path).name}'
        soundfile.write(out_path, _audio, hparams["audio_sample_rate"], 'PCM_16')


if __name__ == '__main__':
    # 工程文件夹名，训练时用的那个
    project_name = "fox_cn"
    model_path = f'./checkpoints/{project_name}/clean_model_ckpt_steps_260000.ckpt'
    config_path = f'./checkpoints/{project_name}/config.yaml'

    # 此脚本为批量导出短音频（30s内）使用，同时生成f0、mel供diffsinger使用。
    # 支持wav文件，放在batch文件夹下，带扩展名
    wav_paths = infer_tool.get_end_file("./batch", "wav")
    trans = -6  # 音高调整，支持正负（半音）
    # 特化专用，开启此项后，仅导出变更音色的units至batch目录，其余项不输出；关闭此项则切换为对接diffsinger的套娃导出模式
    units = True
    # 自适应变调，不懂别开
    auto_key = False
    # 加速倍数
    accelerate = 10

    # 仅支持opencpop标注文件，配合上方移调；使用时自行修改文件名、输出名（带txt后缀）
    if not units:
        trans_opencpop("transcriptions.txt", "res.txt", trans)

    # 下面不动
    os.makedirs("./singer_data", exist_ok=True)
    model = Svc(project_name, config_path, hubert_gpu=True, model_path=model_path)
    count = 0
    for audio_path in wav_paths:
        count += 1
        if os.path.exists(Path(audio_path).with_suffix(".npy")) and units:
            print(f"{audio_path}:units已存在，跳过")
            continue
        run_clip(audio_path, model, trans, accelerate, spk_id=spk_id, auto_key=auto_key, use_crepe=False,
                 units_mode=units)
        print(f"\r\nnum:{count}\r\ntotal process:{round(count * 100 / len(wav_paths), 2)}%\r\n")
