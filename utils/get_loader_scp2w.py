import os
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader

from process.data_fusion import FDDataset_sufr
from process.data_fusion_cefa import FDDataset_cefa
from process.data_fusion_padisi import FDDataset_padisi
from process.data_fusion_wmca import FDDataset_wmca
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataset_scp2w(src1_data, src1_train_num_frames, src2_data, src2_train_num_frames, src3_data, src3_train_num_frames,
                tgt_data, tgt_test_num_frames, batch_size):
    """
    加载源数据和目标数据，创建数据加载器

    参数:
    src1_data (str): 源数据集1的名称
    src1_train_num_frames (int): 源数据集1用于训练的帧数
    src2_data (str): 源数据集2的名称
    src2_train_num_frames (int): 源数据集2用于训练的帧数
    src3_data (str): 源数据集3的名称
    src3_train_num_frames (int): 源数据集3用于训练的帧数
    tgt_data (str): 目标数据集的名称
    tgt_test_num_frames (int): 目标数据集用于测试的帧数
    batch_size (int): 数据加载器的批量大小

    返回:
    tuple: 包含源数据和目标数据的多个数据加载器
    """

    # 创建源数据集1的假样本和真实样本的数据加载器
    src1_train_dataloader_fake = DataLoader(FDDataset_sufr(mode='train', sample_type='fake',image_size=112),
                                            batch_size=batch_size, shuffle=True)
    src1_train_dataloader_real = DataLoader(FDDataset_sufr(mode='train', sample_type='real',image_size=112),
                                            batch_size=batch_size, shuffle=True)
    # 创建源数据集2的假样本和真实样本的数据加载器
    # src2_train_dataloader_fake = DataLoader(FDDataset_padisi(mode='train',sample_type='fake',image_size=112),
    #                                         batch_size=batch_size, shuffle=True)
    # src2_train_dataloader_real = DataLoader(FDDataset_padisi(mode='train',sample_type='real',image_size=112),
    #                                         batch_size=batch_size, shuffle=True)
    src2_train_dataloader_fake = DataLoader(
    ConcatDataset([FDDataset_padisi(mode='train', sample_type='fake', image_size=112) for _ in range(10)]),
    batch_size=batch_size, 
    shuffle=True
)

    src2_train_dataloader_real = DataLoader(
    ConcatDataset([FDDataset_padisi(mode='train', sample_type='real', image_size=112) for _ in range(10)]),
    batch_size=batch_size, 
    shuffle=True
)



    # 创建源数据集3的假样本和真实样本的数据加载器
    src3_train_dataloader_fake = DataLoader(FDDataset_cefa(mode='train',sample_type='fake',dataset_name='4@1',image_size=112),
                                            batch_size=batch_size, shuffle=True)
    src3_train_dataloader_real = DataLoader(FDDataset_cefa(mode='train',sample_type='real',dataset_name='4@1',image_size=112),
                                            batch_size=batch_size, shuffle=True)
    # 创建目标数据集的测试样本的数据加载器
    tgt_dataloader = DataLoader(FDDataset_wmca(mode='val',image_size=112), batch_size=batch_size, shuffle=False)

    return src1_train_dataloader_fake, src1_train_dataloader_real, \
        src2_train_dataloader_fake, src2_train_dataloader_real, \
        src3_train_dataloader_fake, src3_train_dataloader_real, \
        tgt_dataloader
