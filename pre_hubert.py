import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from infer_tools import infer_tool
from preprocessing.hubertinfer import HubertEncoder

# hubert_mode可选——"soft_hubert"、"cn_hubert"
hubert_model = HubertEncoder(hubert_mode='soft_hubert')
# 自动搜索batch文件夹下所有wav文件，可自行更改路径
wav_paths = infer_tool.get_end_file("./batch", "wav")
with tqdm(total=len(wav_paths)) as p_bar:
    p_bar.set_description('Processing')
    for wav_path in wav_paths:
        npy_path = Path(wav_path).with_suffix(".npy")
        if not os.path.exists(npy_path):
            np.save(str(npy_path), hubert_model.encode(wav_path))
        p_bar.update(1)
