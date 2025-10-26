import json
import math
import pandas as pd
import torch
import os
import sys
import shutil

import os
import json
import matplotlib.pyplot as plt

def adjust_learning_rate(optimizer, epoch, init_param_lr, lr_epoch_1, lr_epoch_2):
    """
    调整优化器的学习率

    参数:
    optimizer (torch.optim.Optimizer): 优化器对象
    epoch (int): 当前的训练轮数
    init_param_lr (list): 初始学习率列表，每个参数组对应一个初始学习率
    lr_epoch_1 (int): 第一阶段的学习率变更轮数阈值
    lr_epoch_2 (int): 第二阶段的学习率变更轮数阈值

    返回:
    None
    """
    i = 0
    for param_group in optimizer.param_groups:
        init_lr = init_param_lr[i]  # 获取初始学习率
        i += 1
        if epoch <= lr_epoch_1:
            param_group['lr'] = init_lr * 0.1 ** 0  # 第一阶段，保持初始学习率
        elif epoch <= lr_epoch_2:
            param_group['lr'] = init_lr * 0.1 ** 1  # 第二阶段，学习率降低到初始学习率的10%
        else:
            param_group['lr'] = init_lr * 0.1 ** 2  # 第三阶段，学习率降低到初始学习率的1%

def draw_roc(frr_list, far_list, roc_auc):
    """
    绘制ROC曲线并保存图像和数据

    参数:
    frr_list (list): 假拒绝率列表
    far_list (list): 假接受率列表
    roc_auc (float): AUC值

    返回:
    None
    """
    plt.switch_backend('agg')  # 使用无图形用户界面的后端
    plt.rcParams['figure.figsize'] = (6.0, 6.0)  # 设置图像尺寸
    plt.title('ROC')  # 设置图像标题
    plt.plot(far_list, frr_list, 'b', label='AUC = %0.4f' % roc_auc)  # 绘制ROC曲线并标注AUC值
    plt.legend(loc='upper right')  # 设置图例位置
    plt.plot([0, 1], [1, 0], 'r--')  # 绘制参考线
    plt.grid(ls='--')  # 设置网格线样式
    plt.ylabel('False Negative Rate')  # 设置y轴标签
    plt.xlabel('False Positive Rate')  # 设置x轴标签
    save_dir = './save_results/ROC/'  # 设置保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # 创建目录（如果不存在）
    plt.savefig('./save_results/ROC/ROC.png')  # 保存ROC曲线图像
    file = open('./save_results/ROC/FAR_FRR.txt', 'w')  # 打开文件以保存FAR和FRR数据
    save_json = []
    dict = {}
    dict['FAR'] = far_list  # 保存假接受率数据
    dict['FRR'] = frr_list  # 保存假拒绝率数据
    save_json.append(dict)
    json.dump(save_json, file, indent=4)  # 将数据以JSON格式保存到文件中
    file.close()  # 关闭文件


import os
import json
import math
import pandas as pd


def sample_frames(flag, num_frames, dataset_name):
    """
    从每个视频中采样num_frames帧用于测试
    返回：选定帧的路径和标签

    参数:
    flag (int): 选择的图像类型（0：假图像，1：真实图像，2：所有图像）
    num_frames (int): 每个视频中采样的帧数
    dataset_name (str): 数据集名称

    返回:
    pd.DataFrame: 选定帧的信息（路径、标签、所属视频ID）
    """
    # 设置数据集的根路径
    root_path = '../../data_label/' + dataset_name
    # 根据flag选择标签路径
    if flag == 0:  # 选择假图像
        label_path = root_path + '/fake_label.json'
        save_label_path = root_path + '/choose_fake_label.json'
    elif flag == 1:  # 选择真实图像
        label_path = root_path + '/real_label.json'
        save_label_path = root_path + '/choose_real_label.json'
    else:  # 选择所有真实和假图像
        label_path = root_path + '/all_label.json'
        save_label_path = root_path + '/choose_all_label.json'

    # 加载所有标签的JSON文件
    all_label_json = json.load(open(label_path, 'r'))
    f_sample = open(save_label_path, 'w')
    length = len(all_label_json)
    # 保存每帧的前缀路径
    saved_frame_prefix = '/'.join(all_label_json[0]['photo_path'].split('/')[:-1])
    final_json = []
    video_number = 0
    single_video_frame_list = []
    single_video_frame_num = 0
    single_video_label = 0

    for i in range(length):
        photo_path = all_label_json[i]['photo_path']
        photo_label = all_label_json[i]['photo_label']
        frame_prefix = '/'.join(photo_path.split('/')[:-1])

        # 处理最后一帧
        if i == length - 1:
            photo_frame = int(photo_path.split('/')[-1].split('.')[0])
            single_video_frame_list.append(photo_frame)
            single_video_frame_num += 1
            single_video_label = photo_label

        # 处理已保存的视频帧
        if frame_prefix != saved_frame_prefix or i == length - 1:
            # 对帧列表进行排序
            single_video_frame_list.sort()
            # 计算帧间隔
            frame_interval = math.floor(single_video_frame_num / num_frames)
            # 选择num_frames数量的帧
            for j in range(num_frames):
                dict = {}
                dict['photo_path'] = saved_frame_prefix + '/' + str(
                    single_video_frame_list[6 + j * frame_interval]) + '.png'
                dict['photo_label'] = single_video_label
                dict['photo_belong_to_video_ID'] = video_number
                final_json.append(dict)
            video_number += 1
            saved_frame_prefix = frame_prefix
            single_video_frame_list.clear()
            single_video_frame_num = 0

        # 获取每帧的信息
        photo_frame = int(photo_path.split('/')[-1].split('.')[0])
        single_video_frame_list.append(photo_frame)
        single_video_frame_num += 1
        single_video_label = photo_label

    # 根据flag输出总视频数量
    if flag == 0:
        print("Total video number(fake): ", video_number, dataset_name)
    elif flag == 1:
        print("Total video number(real): ", video_number, dataset_name)
    else:
        print("Total video number(target): ", video_number, dataset_name)

    # 将选择的帧信息保存到JSON文件
    json.dump(final_json, f_sample, indent=4)
    f_sample.close()

    # 读取并返回选择的帧信息
    f_json = open(save_label_path)
    sample_data_pd = pd.read_json(f_json)
    return sample_data_pd


class AverageMeter(object):
    """计算并存储平均值和当前值的类"""

    def __init__(self):
        """初始化方法，调用reset方法重置所有参数"""
        self.reset()

    def reset(self):
        """重置所有参数，将当前值、平均值、总和和计数器都设为0"""
        self.val = 0  # 当前值
        self.avg = 0  # 平均值
        self.sum = 0  # 总和
        self.count = 0  # 计数器

    def update(self, val, n=1):
        """
        更新方法，更新当前值、总和、计数器和平均值

        参数:
        val (float): 本次更新的值
        n (int): 本次更新的计数（默认为1）
        """
        self.val = val  # 更新当前值
        self.sum += val * n  # 更新总和
        self.count += n  # 更新计数器
        self.avg = self.sum / self.count  # 计算平均值


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


import os
import sys


def mkdirs(checkpoint_path, best_model_path, logs):
    """
    创建必要的目录，如果目录不存在则创建它们。

    参数:
    checkpoint_path (str): 用于保存检查点的目录路径。
    best_model_path (str): 用于保存最佳模型的目录路径。
    logs (str): 用于保存日志的目录路径。
    """
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    if not os.path.exists(logs):
        os.mkdir(logs)


def time_to_str(t, mode='min'):
    """
    将时间转换为字符串格式，表示为分钟或秒。

    参数:
    t (int): 时间，单位为秒。
    mode (str): 转换模式，'min' 表示转换为分钟，'sec' 表示转换为秒。

    返回:
    str: 转换后的时间字符串。
    """
    if mode == 'min':
        t = int(t) / 60
        hr = t // 60
        min = t % 60
        return '%2d hr %02d min' % (hr, min)
    elif mode == 'sec':
        t = int(t)
        min = t // 60
        sec = t % 60
        return '%2d min %02d sec' % (min, sec)
    else:
        raise NotImplementedError


class Logger(object):
    """
    日志记录器类，用于同时将日志写入终端和文件。
    """

    def __init__(self):
        self.terminal = sys.stdout
        self.file = None

    def open(self, file, mode=None):
        """
        打开一个日志文件。

        参数:
        file (str): 日志文件路径。
        mode (str): 文件打开模式，默认是 'w' (写入)。
        """
        if mode is None:
            mode = 'w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        """
        写入日志信息到终端和文件。

        参数:
        message (str): 要写入的日志信息。
        is_terminal (int): 是否将日志写入终端，1 表示写入，0 表示不写入。
        is_file (int): 是否将日志写入文件，1 表示写入，0 表示不写入。
        """
        if '\r' in message:
            is_file = 0
        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        """
        刷新方法，为了兼容 Python 3 而设置。
        此方法在执行 flush 命令时不执行任何操作。
        你可以在这里指定一些额外的行为。
        """
        pass


def save_checkpoint(save_list, is_best, model, gpus, checkpoint_path, best_model_path, filename='_checkpoint.pth.tar'):
    epoch = save_list[0]
    valid_args = save_list[1]
    best_model_HTER = round(save_list[2], 5)
    best_model_ACC = save_list[3]
    best_model_ACER = save_list[4]
    threshold = save_list[5]
    if(len(gpus) > 1):
        old_state_dict = model.state_dict()
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in old_state_dict.items():
            flag = k.find('.module.')
            if (flag != -1):
                k = k.replace('.module.', '.')
            new_state_dict[k] = v
        state = {
            "epoch": epoch,
            "state_dict": new_state_dict,
            "valid_arg": valid_args,
            "best_model_EER": best_model_HTER,
            "best_model_ACER": best_model_ACER,
            "best_model_ACC": best_model_ACC,
            "threshold": threshold
        }
    else:
        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "valid_arg": valid_args,
            "best_model_EER": best_model_HTER,
            "best_model_ACER": best_model_ACER,
            "best_model_ACC": best_model_ACC,
            "threshold": threshold
        }
    filepath = checkpoint_path + filename
    torch.save(state, filepath)
    # just save best model
    if is_best:
        shutil.copy(filepath, best_model_path + 'model_best_' + str(best_model_HTER) + '_' + str(epoch) + '.pth.tar')

def zero_param_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()