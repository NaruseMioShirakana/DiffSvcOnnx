# Diff-SVC

Singing Voice Conversion via diffusion model

## 本仓库为diff-svc fork重构版，新增辅助脚本、新hubert等，请自行评估并承担风险

> 建议使用稳定版：[diff-svc](https://github.com/prophesier/diff-svc) \
> 项目教程在doc文件夹下，此魔改版问题请勿在原项目频道、discord等询问。 \
> 同参数下，中文hubert所需训练步数约为soft hubert的1.5~2倍，不建议新手使用

## updates:

> 2023.01.20 重构目录，精简代码，去除多层继承@小狼 \
> 2023.01.18 配置文件改为级联，仅需修改config_nsf、config_ms（二选一）即可预处理@小狼 \
> 2023.01.16 增加多说话人支持(config_ms.yaml)，预处理代码参考diffsinger修改@小狼 \
> 2023.01.09 \
> 新增select.py筛选数据集音域（数据量足够时，删去重复音域部分，加快高低音收敛）\
> 删除24k的pe、hifigan等依赖，删除pitch cwt模式，infer复用预处理部分代码@小狼
> 2023.01.07 预处理新增f0_static超参统计音域，新增自适应变调功能(需f0_static，旧模型config可用data_static添加此超参)@小狼
> 2023.01.05 \
> 取消24k采样率、pe支持，删减部分参数、doc新增特化教程；batch.py支持特化、套娃两种模式的导出；\
> pre_hubert为分步预处理（4g及以下内存预处理使用）；data_static为数据集音域统计（仅供参考）；中文hubert所需依赖fairseq请自行安装@小狼\
> 2023.01.01 更新切片机v2、取消切片缓存，简化部分infer流程；取消vec支持、增加中文hubert（仅base模型，1.1g左右）@小狼\
> 2022.12.17 新增pre_check检测环境、数据@深夜诗人; 改进simplify精简模型@九尾玄狐; 监修代码@小狼\
> 2022.12.16 修复推理时hubert模型重复加载的问题@小狼\
> 2022.12.04 44.1kHz声码器开放申请，正式提供对44.1kHz的支持\
> 2022.11.28 增加了默认打开的no_fs2选项，可优化部分网络，提升训练速度、缩减模型体积，对于未来新训练的模型有效\
> 2022.11.23 \
> 修复了一个重大bug，曾导致可能将用于推理的原始gt音频转变采样率为22.05kHz,对于由此造成的影响我们表示十分抱歉，请务必检查自己的测试音频，并使用更新后的代码\
> 2022.11.22 修复了很多bug，其中有几个影响推理效果重大的bug\
> 2022.11.20 增加对推理时多数格式的输入和保存，无需手动借助其他软件转换\
> 2022.11.13 修正中断后读取模型的epoch/steps显示问题，添加f0处理的磁盘缓存，添加实时变声推理的支持文件\
> 2022.11.11 修正切片时长误差，补充对44.1khz的适配, 增加对contentvec的支持\
> 2022.11.04 添加梅尔谱保存功能\
> 2022.11.02 整合新声码器代码，更新parselmouth算法\
> 2022.10.29 整理推理部分，添加长音频自动切片功能。\
> 2022.10.28 将hubert的onnx推理迁移为torch推理，并整理推理逻辑。\
<font color=#FFA500>如原先下载过onnx的hubert模型需重新下载并替换为pt模型</font>，config不需要改，目前可以实现1060
> 6G显存的直接GPU推理与预处理，详情请查看文档。\
> 2022.10.27 更新依赖文件，去除冗余依赖。\
> 2022.10.27 修复了一个严重错误，曾导致在gpu服务器上hubert仍使用cpu推理，速度减慢3-5倍，影响预处理与推理，不影响训练\
> 2022.10.26 修复windows上预处理数据在linux上无法使用的问题，更新部分文档\
> 2022.10.25 编写推理/训练详细文档，修改整合部分代码，增加对ogg格式音频的支持(无需与wav区分，直接使用即可)\
> 2022.10.24 支持对自定义数据集的训练，并精简代码\
> 2022.10.22 完成对opencpop数据集的训练并创建仓库

## 注意事项/Notes：

> 本项目是基于学术交流目的建立，并非为生产环境准备，不对由此项目模型产生的任何声音的版权问题负责。\
> 如将本仓库代码二次分发，或将由此项目产出的任何结果公开发表(包括但不限于视频网站投稿)，请注明原作者及代码来源(此仓库)。\
> 如果将此项目用于任何其他企划，请提前联系并告知本仓库作者,十分感谢。\
> This project is established for academic exchange purposes and is not intended for production environments. We are not
> responsible for any copyright issues arising from the sound produced by this project's model. \
> If you redistribute the code in this repository or publicly publish any results produced by this project (including
> but
> not limited to video website submissions), please indicate the original author and source code (this repository). \
> If you use this project for any other plans, please contact and inform the author of this repository in advance. Thank
> you very much.

## 推理/inference：

> infer.py

## 预处理/preprocessing:

```
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 python preprocessing\svc_binarizer.py --config configs/config_nsf.yaml
```

## 训练/training:

```
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/config_nsf.yaml --exp_name [your project name] --reset 
```

> 详细训练过程和各种参数介绍请查看[推理与训练说明](./doc/train_and_inference.markdown) \
> 中文hubert与特化教程[特化教程](./doc/advanced_skills.markdown) \

## Acknowledgements

项目基于[diffsinger](https://github.com/MoonInTheRiver/DiffSinger)、[diffsinger(openvpi维护版)](https://github.com/openvpi/DiffSinger)、[soft-vc](https://github.com/bshall/soft-vc)
开发.\
同时也十分感谢openvpi成员在开发训练过程中给予的帮助。\
This project is based
on [diffsinger](https://github.com/MoonInTheRiver/DiffSinger), [diffsinger (openvpi maintenance version)](https://github.com/openvpi/DiffSinger),
and [soft-vc](https://github.com/bshall/soft-vc). We would also like to thank the openvpi members for their help during
the development and training process. \
> 注意：此项目与同名论文[DiffSVC](https://arxiv.org/abs/2105.13871)无任何联系，请勿混淆！\
> Note: This project has no connection with the paper of the same name [DiffSVC](https://arxiv.org/abs/2105.13871),
> please
> do not confuse them!
> 音频切片参考[audio-slicer](https://github.com/openvpi/audio-slicer)