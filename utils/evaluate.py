from utils.utils import AverageMeter, accuracy
from utils.statistic import get_EER_states, get_HTER_at_thr, calculate, calculate_threshold
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


def eval(valid_dataloader, model, norm_flag):
    criterion = nn.CrossEntropyLoss()  # 定义损失函数为交叉熵损失
    valid_losses = AverageMeter()  # 创建一个记录损失的 AverageMeter 对象
    valid_top1 = AverageMeter()  # 创建一个记录 top1 准确率的 AverageMeter 对象
    prob_dict = {}  # 用于存储每个视频的预测概率
    label_dict = {}  # 用于存储每个视频的标签
    model.eval()  # 将模型设置为评估模式
    output_dict_tmp = {}  # 用于临时存储每个视频的模型输出
    target_dict_tmp = {}  # 用于临时存储每个视频的目标标签

    with torch.no_grad():  # 禁用梯度计算
        for iter, (input, target, videoID) in enumerate(valid_dataloader):
            # print("input.shape_aaa",input.shape)
            input = Variable(input).cuda()  # 将输入数据转换为 GPU 变量
            # print("input.shape",input.shape)
            # print("target.shape",target.shape)
            # print(videoID)
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()  # 将目标标签转换为 GPU 变量并确保是 long 类型
            cls_out, feature,a = model(input, norm_flag)  # 获取模型的分类输出和特征
            prob = F.softmax(cls_out, dim=1).cpu().data.numpy()[:, 1]  # 计算 softmax 概率并提取第二类的概率
            label = target.cpu().data.numpy()  # 将标签转换为 numpy 数组
            # videoID = videoID.cpu().data.numpy()  # 将视频ID转换为 numpy 数组
            videoID = np.array(videoID)  # 将视频ID转换为 numpy 数组

            for i in range(len(prob)):
                if videoID[i] in prob_dict.keys():
                    # 如果 videoID 已经在字典中，追加新值
                    prob_dict[videoID[i]].append(prob[i])
                    label_dict[videoID[i]].append(label[i])
                    output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 2))
                    target_dict_tmp[videoID[i]].append(target[i].view(1))
                else:
                    # 如果 videoID 不在字典中，创建新条目
                    prob_dict[videoID[i]] = []
                    label_dict[videoID[i]] = []
                    prob_dict[videoID[i]].append(prob[i])
                    label_dict[videoID[i]].append(label[i])
                    output_dict_tmp[videoID[i]] = []
                    target_dict_tmp[videoID[i]] = []
                    output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 2))
                    target_dict_tmp[videoID[i]].append(target[i].view(1))

    prob_list = []
    label_list = []
    for key in prob_dict.keys():
        avg_single_video_prob = sum(prob_dict[key]) / len(prob_dict[key])  # 计算每个视频的平均预测概率
        avg_single_video_label = sum(label_dict[key]) / len(label_dict[key])  # 计算每个视频的平均标签
        prob_list = np.append(prob_list, avg_single_video_prob)  # 将平均预测概率添加到列表中
        label_list = np.append(label_list, avg_single_video_label)  # 将平均标签添加到列表中

        # 计算每个视频的损失和准确率
        avg_single_video_output = sum(output_dict_tmp[key]) / len(output_dict_tmp[key])  # 计算每个视频的平均输出
        # print("avg_single_video_output",avg_single_video_output)
        avg_single_video_target = sum(target_dict_tmp[key]) / len(target_dict_tmp[key])  # 计算每个视频的平均目标
        avg_single_video_target = avg_single_video_target.long()

        # print("avg_single_video_output",avg_single_video_target)
        loss = criterion(avg_single_video_output, avg_single_video_target)  # 计算损失
        acc_valid = accuracy(avg_single_video_output, avg_single_video_target, topk=(1,))  # 计算 top1 准确率
        valid_losses.update(loss.item())  # 更新损失
        valid_top1.update(acc_valid[0])  # 更新 top1 准确率

    auc_score = roc_auc_score(label_list, prob_list)  # 计算 AUC 分数
    cur_EER_valid, threshold, _, _ = get_EER_states(prob_list, label_list)  # 计算 EER（等错误率）和阈值
    ACC_threshold = calculate_threshold(prob_list, label_list, threshold)  # 根据阈值计算准确率
    cur_HTER_valid = get_HTER_at_thr(prob_list, label_list, threshold)  # 计算 HTER（半总错误率）

    return [valid_losses.avg, valid_top1.avg, cur_EER_valid, cur_HTER_valid, auc_score, threshold, ACC_threshold * 100]
    # 返回验证损失的平均值、top1 准确率的平均值、EER、HTER、AUC 分数、阈值和阈值下的准确率
