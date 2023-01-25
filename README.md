# Diff-SVC

Singing Voice Conversion via diffusion model

---
## 本仓库为diff-svc fork重构版，本仓库主要为Onnx相关内容

> 原始地址：[diff-svc](https://github.com/prophesier/diff-svc) 
---
## 使用方法

> 1、创建checkpoints文件夹，在其中创建另一个文件夹作为项目路径，在其中放置模型和配置文件，分别改名为model.ckpt与config.yaml

> 2、打开infer.py，将project_name改为你的项目文件夹名

> 3、创建raw文件夹，里面放一个时长小于5秒的音频，将infer.py中的file_names改为该音频文件名

> 4、修改accelerate，指定一个加速倍数（建议100左右，除非你想体验CPU推理一个音频几小时）

> 5、运行infer.py，等待输出onnx