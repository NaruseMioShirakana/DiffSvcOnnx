import os
import re

import yaml

solutions = {'yaml': 'yaml路径不正确，可能导致yaml污染或预处理失败\r\n',
             'hubert': 'hubert不存在，请到群文件里下载hubert_torch.zip并解压到checkpoints文件夹下\r\n',
             'raw_data_dir': '数据集目录与yaml不匹配，请检查数据集目录与yaml内"raw_data_dir:"栏是否匹配\r\n',
             'vocoder': "与yaml匹配的声码器不存在，24k声码器请到群文件里下载basics.zip并解压到checkpoints文件夹下\r\n44.1k声码器请到“https://github.com/openvpi/vocoders”github发布页下载并解压到checkpoints下\r\n如已下载并解压，请核对声码器文件名与yaml内“vocoder_ckpt:”栏的声码器文件名与checkpoints文件夹下声码器文件夹内的声码器文件名是否匹配\r\n",
             'torch': '你未安装torch三件套或torch三件套异常,解决方法见语雀安装torch相关，\r\n复制链接粘贴到浏览器即可直达相关网页:\r\n通用命令安装torch：https://www.yuque.com/jiuwei-nui3d/qng6eg/sc8ivoge8vww4lu6#9mQgt\r\nwindows下手动安装torch：https://www.yuque.com/jiuwei-nui3d/qng6eg/ea0ntd\r\n',
             'urllib.parse': '载入urllib.parse失败,解决方法见语雀常见错误①，\r\n复制链接粘贴到浏览器即可直达相关网页:https://www.yuque.com/jiuwei-nui3d/qng6eg/gdpi5orf3niv9mwb#SyTom\r\n',
             'utils.hparams': '载入utils.hparams失败,解决方法见语雀常见错误③，\r\n复制链接粘贴到浏览器即可直达相关网页:https://www.yuque.com/jiuwei-nui3d/qng6eg/abaxpwozc2h5yltt#MOddD\r\n',
             'config_path': 'config_path路径，即yaml路径格式不正确，应为：training/xxxx.yaml\r\n'}


def get_end_file(dir_path, end):
    file_lists = []
    for root, dirs, files in os.walk(dir_path):
        files = [f for f in files if f[0] != '.']
        dirs[:] = [d for d in dirs if d[0] != '.']
        for f_file in files:
            if f_file.endswith(end):
                file_lists.append(os.path.join(root, f_file).replace("\\", "/"))
    return file_lists


def scan(path):
    model_str = ""
    path_lists = get_end_file(path, "yaml")
    for i in range(0, len(path_lists)):
        if re.search(u'[\u4e00-\u9fa5]', path_lists[i]):
            print(f'{path_lists[i]}:中文路径！此项跳过')
            continue
        model_str += f"{i}:{path_lists[i]}\r\n"
        if (i + 1) % 5 == 0:
            print(f"{model_str}")
            model_str = ""
    if len(path_lists) % 5 != 0:
        print(model_str)
    return path_lists


# 检测文件夹大小的函数
def get_dir_size(path, size=0):
    for root, dirs, files in os.walk(path):
        for f in files:
            size += os.path.getsize(os.path.join(root, f))
    return size


# 意义不明的try_except
def try_except():
    print("请等待10秒")
    res_str = ""
    try:
        import torch
        import torchvision
        import torchaudio
        print('成功加载torch')
        cuda = f"cuda vision: {torch.cuda_version}" if torch.cuda.is_available() else "cuda不存在或版本不匹配，请查阅相关资料自行安装"
        print(cuda)
    except Exception as e:
        res_str += solutions['torch']

    try:
        from urllib.parse import quote
        print('成功载入urllib.parse')
    except Exception as e:
        res_str += solutions['urllib.parse']

    try:
        from utils.hparams import set_hparams, hparams
        print('成功载入utils.hparams')
    except Exception as e:
        res_str += solutions['utils.hparams']
    if res_str:
        print("\r\n*=====================\r\n", "错误及解决方法:\r\n", res_str)


def test_part(test):
    res_str = ""
    print("\r\n*=====================")
    for k, v in test.items():
        if isinstance(v, list):
            for i in v:
                if os.path.exists(i):
                    print(f"{k}-{i}:  通过" + (
                        "，绝对路径只能在当前平台运行，更换平台训练请使用相对路径" if os.path.isabs(i) else ""))
        elif os.path.exists(v):
            print(
                f"{k}:  通过" + ("，绝对路径只能在当前平台运行，更换平台训练请使用相对路径" if os.path.isabs(v) else ""))
        else:
            print(f"{k}:  不通过")
            res_str += f"{k}:{solutions[k]}\r\n"
    if res_str:
        print("\r\n解决方法:\r\n", res_str)
    else:
        return True


if __name__ == '__main__':
    print("选择:")
    print("0.环境检测")
    print("1.配置文件检测")
    f = int(input("请输入选项:"))
    if f == 0:
        # 调用try函数
        try_except()
    elif f == 1:
        path_list = scan("./configs")
        a = input("请输入选项:")
        project_path = path_list[int(a)]
        with open(project_path, "r") as f:
            data = yaml.safe_load(f)
        with open("./configs/base.yaml", "r") as f:
            base = yaml.safe_load(f)
        test_model = {'yaml': data["config_path"], 'hubert': data["hubert_path"],
                      'raw_data_dir': data["raw_data_dir"], 'vocoder': base["vocoder_ckpt"],
                      'config_path': data["config_path"]}
        try_except()
        yaml_path = data["config_path"]
        model_name = data["binary_data_dir"].split("/")[-1]
        if test_part(test_model):
            if get_dir_size(data["binary_data_dir"]) > 100 * 1024 ** 2:
                print("\r\ntrain.data通过初步检测（不排除数据集制作时的失误）")
                print("\r\n*====================="
                      "\r\n### 训练"
                      "\r\ncd进入diff-svc的目录下执行以下命令："
                      "\r\n*====================="
                      "\r\n# windows，**使用cmd窗口**"
                      "\r\nset CUDA_VISIBLE_DEVICES=0"
                      f"\r\npython run.py --config {yaml_path} --exp_name {model_name} --reset"
                      "\r\n*====================="
                      "\r\n# linux"
                      f"\r\nCUDA_VISIBLE_DEVICES=0 python run.py --config {yaml_path} --exp_name {model_name} --reset"
                      "\r\n*=====================")
            else:
                print("\r\n未进行预处理或预处理错误，请参考语雀教程：https://www.yuque.com/jiuwei-nui3d/qng6eg")
                print("\r\n*====================="
                      "\r\n### 数据预处理"
                      "\r\ncd进入diff-svc的目录下执行以下命令："
                      "\r\n*====================="
                      "\r\n# windows，**使用cmd窗口**"
                      "\r\nset PYTHONPATH=."
                      "\r\nset CUDA_VISIBLE_DEVICES=0"
                      f"\r\npython preprocessing/binarize.py --config {yaml_path}"
                      "\r\n*====================="
                      "\r\n# linux"
                      "\r\nexport PYTHONPATH=."
                      f"\r\nCUDA_VISIBLE_DEVICES=0 python preprocessing\svc_binarizer.py --config {yaml_path}"
                      "\r\n*=====================")
                print("预处理完请重新运行此脚本选项1，届时提供训练命令")
    else:
        print("请依据以上提示解决问题后，重新运行此脚本")
        exit()
