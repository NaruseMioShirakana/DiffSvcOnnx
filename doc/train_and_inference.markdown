# Diff-SVC(train/inference by yourself)

## 基于原版教程修改

## 0.环境配置

```
pip install -r requirements.txt
```

## 1.推理

> 使用根目录下的infer.py\
> 在第一个block中修改如下参数：

```
config_path='checkpoints压缩包中config.yaml的位置'
如'./checkpoints/nyaru/config.yaml'
config和checkpoints是一一对应的，请不要使用其他config

project_name='这个项目的名称'
如'nyaru'

model_path='ckpt文件的全路径'
如'./checkpoints/nyaru/model_ckpt_steps_112000.ckpt'

hubert_gpu=True
推理时是否使用gpu推理hubert(模型中的一个模块)，不影响模型的其他部分
目前版本已大幅减小hubert的gpu占用，在1060 6G显存下可完整推理，不需要关闭了。
另外现已支持长音频自动切片功能(ipynb和infer.py均可)，超过30s的音频将自动在静音处切片处理，感谢@小狼的代码

```

### 可调节参数：

```
file_names=["逍遥仙","xxx"]#传入音频的路径，默认在文件夹raw中

use_crepe=True 
#crepe是一个F0算法，效果好但速度慢，改成False会使用效果稍逊于crepe但较快的parselmouth算法

thre=0.05
#crepe的噪声过滤阈值，源音频干净可适当调大，噪音多就保持这个数值或者调小，前面改成False后这个参数不起作用

pndm_speedup=20
#推理加速算法倍数，默认是1000步，这里填成10就是只使用100步合成，是一个中规中矩的数值，这个数值可以高到50倍(20步合成)没有明显质量损失，再大可能会有可观的质量损失,注意如果下方开启了use_gt_mel, 应保证这个数值小于add_noise_step，并尽量让其能够整除

key=0
#变调参数，默认为0(不是1!!)，将源音频的音高升高key个半音后合成，如男声转女生，可填入8或者12等(12就是升高一整个8度)

wav_gen='yyy.wav'#输出音频的路径，默认在项目根目录中，可通过改变扩展名更改保存文件类型
```

## 2.数据预处理与训练

### 2.1 准备数据

> 目前支持wav格式和ogg格式的音频数据，采样率最好高于24kHz，程序会自动处理采样率和声道问题。采样率不可低于16kHz（一般不会的）\
> 音频需要切片为5-15s为宜的短音频，长度没有具体要求，但不宜过长过短。音频需要为纯目标人干声，不可以有背景音乐和其他人声音，最好也不要有过重的混响等。若经过去伴奏等处理，请尽量保证处理后的音频质量。\
> 单人训练复制config_nsf.yaml修改，总时长尽量保证在3h或以上，不需要额外任何标注。

### 2.2 修改超参数配置

> 首先请备份一份config_nsf.yaml（configs文件夹下），然后修改它\
> 多人训练复制config_ms.yaml修改 \
> 可能会用到的参数如下(以工程名为nyaru为例):

```
K_step: 1000
#diffusion过程总的step,建议不要修改

binary_data_dir: data/binary/nyaru
预处理后数据的存放地址:需要将后缀改成工程名字

config_path: configs/config_nsf.yaml
你要使用的这份yaml自身的地址，由于预处理过程中会写入数据，所以这个地址务必修改成将要存放这份yaml文件的完整路径

choose_test_manually: false
手动选择测试集，默认关闭，自动随机抽取5条音频作为测试集。
如果改为ture，请在test_prefixes:中填入测试数据的文件名前缀，程序会将以对应前缀开头的文件作为测试集
这是个列表，可以填多个前缀，如：
test_prefixes:
- test
- aaaa
- 5012
- speaker1024
重要：测试集*不可以*为空，为了不产生意外影响，建议尽量不要手动选择测试集

endless_ds:False
如果你的数据集过小，每个epoch时间很短，请将此项打开，将把正常的1000epoch作为一个epoch计算

hubert_path: checkpoints/hubert/hubert_soft.pt
hubert模型的存放地址，确保这个路径是对的，一般解压checkpoints包之后就是这个路径不需要改,现已使用torch版本推理
hubert_gpu:True
是否在预处理时使用gpu运行hubert(模型的一个模块)，关闭后使用cpu，但耗时会显著增加。另外模型训练完推理时hubert是否用gpu是在inference中单独控制的，不受此处影响。目前hubert改为torch版后已经可以做到在1060 6G显存gpu上进行预处理，与直接推理1分钟内的音频不超出显存限制，一般不需要关了。

lr: 0.0008
#初始的学习率:这个数字对应于88的batchsize(80g显存)，如果batchsize更小，可以调低这个数值一些

decay_steps: 20000
每20000步学习率衰减为原来的一半，如果batchsize比较小，请调大这个数值

#对于30-40左右的batchsize(30g显存)，推荐lr=0.0004，decay_steps=40000

max_frames: 42000
max_input_tokens: 6000
max_sentences: 88
max_tokens: 128000
#batchsize是由这几个参数动态算出来的，如果不太清楚具体含义，可以只改动max_sentences这个参数，填入batchsize的最大限制值，以免炸显存

raw_data_dir: data/raw/nyaru
#存放预处理前原始数据的位置，请将原始wav数据放在这个目录下，内部文件结构无所谓，会自动解构

residual_channels: 384
residual_layers: 20
#控制核心网络规模的一组参数，越大参数越多炼的越慢，但效果不一定会变好，大一点的数据集可以把第一个改成512。这个可以自行实验效果，不过不了解的话尽量不动。

use_crepe: true
#在数据预处理中使用crepe提取F0,追求效果请打开，追求速度可以关闭

val_check_interval: 2000
#每2000steps推理测试集并保存ckpt

vocoder_ckpt:checkpoints/nsf_hifigan/model
#对应声码器的文件名, 注意不要填错

work_dir: checkpoints/nyaru
#修改后缀为工程名
```

> 其他的参数如果你不知道它是做什么的，请不要修改，即使你看着名称可能以为你知道它是做什么的。

### 2.3 数据预处理

在diff-svc的目录下执行以下命令：\
#windows

```
set PYTHONPATH=.
set CUDA_VISIBLE_DEVICES=0 
python preprocessing\svc_binarizer.py --config configs/config_nsf.yaml
```

#linux

```
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 python preprocessing\svc_binarizer.py --config configs/config_nsf.yaml
```

对于预处理，@小狼准备了一份可以分段处理hubert和其他特征的代码，如果正常处理显存不足，可以修改后使用 \
pre_hubert.py, 然后再运行正常的指令，能够识别提前处理好的hubert特征

### 2.4 训练

#windows

```
set CUDA_VISIBLE_DEVICES=0 
python run.py --config configs/config.yaml --exp_name nyaru --reset  
```

#linux

```
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/config.yaml --exp_name nyaru --reset 
```

> 需要将exp_name改为你的工程名，并修改config路径，请确保和预处理使用的是同一个config文件\
*重要*
>
：训练完成后，若之前不是在本地数据预处理，除了需要下载对应的ckpt文件，也需要将config文件下载下来，作为推理时使用的config，不可以使用本地之前上传上去那份。因为预处理时会向config文件中写入内容。推理时要保持使用的config和预处理使用的config是同一份。

### 2.5 可能出现的问题：

> 2.5.1 'Upsample' object has no attribute 'recompute_scale_factor'\
> 此问题发现于cuda11.3对应的torch中，若出现此问题,请通过合适的方法(如ide自动跳转等)
> 找到你的python依赖包中的torch.nn.modules.upsampling.py文件(
> 如conda环境中为conda目录\envs\环境目录\Lib\site-packages\torch\nn\modules\upsampling.py)，修改其153-154行

```
return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners,recompute_scale_factor=self.recompute_scale_factor)
```

> 改为

```
return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)
# recompute_scale_factor=self.recompute_scale_factor)
```

> 2.5.2 no module named 'utils'\
> 请在你的运行环境(如colab笔记本)中以如下方式设置:

```
import os
os.environ['PYTHONPATH']='.'
!CUDA_VISIBLE_DEVICES=0 python preprocessing/binarize.py --config training/config.yaml
```

注意一定要在项目文件夹的根目录中执行
> 2.5.3 cannot load library 'libsndfile.so'\
> 可能会在linux环境中遇到的错误,请执行以下指令

```
apt-get install libsndfile1 -y
```

> 2.5.4 cannot load import 'consume_prefix_in_state_dict_if_present'\
> torch版本过低，请更换高版本torch

> 2.5.5 预处理数据过慢\
> 检查是否在配置中开启了use_crepe，将其关闭可显著提升速度。\
> 检查配置中hubert_gpu是否开启。
