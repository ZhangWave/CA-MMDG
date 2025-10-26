import json
import math

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch
import os
import sys
import shutil
import numpy as np


def image2cols(image, patch_size, stride):
    import numpy as np
    if len(image.shape) == 2:
        imhigh, imwidth = image.shape
        imch = 1
    elif len(image.shape) == 3:
        imhigh, imwidth, imch = image.shape
    else:
        raise ValueError("Unsupported image shape.")

    range_y = np.arange(0, imhigh - patch_size[0], stride)
    range_x = np.arange(0, imwidth - patch_size[1], stride)


    if range_y[-1] != imhigh - patch_size[0]:
        range_y = np.append(range_y, imhigh - patch_size[0])

    if range_x[-1] != imwidth - patch_size[1]:
        range_x = np.append(range_x, imwidth - patch_size[1])

    sz = len(range_y) * len(range_x)

    if imch == 1:
        res = np.zeros((sz, patch_size[0], patch_size[1]))
    else:
        res = np.zeros((sz, patch_size[0], patch_size[1], imch))

    index = 0
    for y in range_y:
        for x in range_x:
            patch = image[y:y + patch_size[0], x:x + patch_size[1]]
            res[index] = patch
            index = index + 1
    return res

def col2image(coldata, imsize, stride):
    """
    coldata: 使用image2cols得到的数据
    imsize:原始图像的宽和高，如(321, 481)
    stride:图像切分时的步长，如10
    """
    patch_size = coldata.shape[1:3]
    if len(coldata.shape) == 3:
        ## 初始化灰度图像
        res = np.zeros((imsize[0], imsize[1]))
        w = np.zeros(((imsize[0], imsize[1])))
    if len(coldata.shape) == 4:
        ## 初始化RGB图像
        res = np.zeros((imsize[0], imsize[1], 3))
        w = np.zeros(((imsize[0], imsize[1], 3)))
    range_y = np.arange(0, imsize[0] - patch_size[0], stride)
    range_x = np.arange(0, imsize[1] - patch_size[1], stride)
    if range_y[-1] != imsize[0] - patch_size[0]:
        range_y = np.append(range_y, imsize[0] - patch_size[0])
    if range_x[-1] != imsize[1] - patch_size[1]:
        range_x = np.append(range_x, imsize[1] - patch_size[1])
    index = 0
    for y in range_y:
        for x in range_x:
            res[y:y + patch_size[0], x:x + patch_size[1]] = res[y:y + patch_size[0], x:x + patch_size[1]] + coldata[
                index]
            w[y:y + patch_size[0], x:x + patch_size[1]] = w[y:y + patch_size[0], x:x + patch_size[1]] + 1
            index = index + 1
    return res / w


def image2cols_batch(image, patch_size, stride):
    batch = image.shape[0]
    sz = (image.shape[1] / patch_size[0]) * (image.shape[2] / patch_size[1])
    Res = np.zeros((batch, int(sz), patch_size[0], patch_size[1], 3))
    for i in range(batch):
        current_image = image[i]
        Res[i] = image2cols(current_image, patch_size, stride)
    return Res


def col2image_batch(coldata, imsize, stride):
    batch = coldata.shape[0]
    final_image = np.zeros((batch, imsize[0], imsize[1], 3))
    for i in range(batch):
        final_image[i] = col2image(coldata[i], imsize, stride)
    return final_image


# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
#
# # 加载一张图像并转换成numpy数组
# image_path = r'F:\BaiduNetdiskDownload\padisi_USC_FACE_MULTI_SPECTRAL_preprocessed\padisi_face\ID_000f8ecbd2c1e2c270b021eebae7ebb75c7ed300_6f1a0860_FACE_FROWN_WITHOUT_GLASS\color.png'  # 替换为你的图像路径
#
# my_image = np.array(Image.open(image_path))
# if my_image.shape[2] == 4:
#     my_image = my_image[:, :, :3]
# print(my_image.shape)
# # 图像块的尺寸
# patch_size = (50, 50)  # 例如，我们想要切分成50x50的图像块
#
# # 切分图像块时的步长
# stride = 50  # 图像块之间以25像素的步长移动
#
# # 调用image2cols函数
# image_blocks = image2cols(my_image, patch_size, stride)
# print(image_blocks.shape)
#
# # 打印切分结果
# print(f"共切分出 {image_blocks.shape[0]} 个图像块。")
# # 显示第一个图像块
# first_block = image_blocks[7]
# if first_block.ndim == 3:
#     plt.imshow(first_block.astype(np.uint8))
# else:  # 如果是灰度图像
#     plt.imshow(first_block.astype(np.uint8), cmap='gray')
# plt.title("第一个图像块")
# plt.show()
# print(image_blocks.shape)
#
#
# reconstructed_image = col2image(image_blocks, my_image.shape[:2], stride)
# # 显示重建图像
# plt.subplot(1, 2, 2)
# plt.imshow(reconstructed_image.astype(np.uint8))
# plt.title("重建图像")
# plt.show()

import torch
from torch.utils.data import DataLoader
from process.data_patch_s import FDDataset_sufr
from process.data_patch_p import FDDataset_padisi
from process.data_patch_c import FDDataset_cefa
from process.data_patch_w import FDDataset_wmca
from torch.utils.data import ConcatDataset
# 创建一个目录用于保存处理后的图像
output_dir = r'F:\BaiduNetdiskDownload\patchmix\scp'
os.makedirs(output_dir, exist_ok=True)

# 假设你有一个训练数据集 train_dataset 和一个 DataLoader train_dataloader
# train_dataset1 = FDDataset_padisi(mode='train', image_size=128,
#                                         fold_index=-1)

padisi_datasets = []

num_padisi_datasets = 4  # 指定生成的padisi数据集数量
for i in range(num_padisi_datasets):
    padisi_dataset = FDDataset_padisi(mode='train', image_size=128,
                                              fold_index=-1)
    padisi_datasets.append(padisi_dataset)
padisi_concat_dataset = ConcatDataset(padisi_datasets)

train_dataset1=FDDataset_sufr(mode='train', image_size=128,
                                        fold_index=-1)

train_dataset2 = FDDataset_cefa(mode='train', image_size=128, dataset_name='4@1',
                                        fold_index=-1)

train_dataset = ConcatDataset([train_dataset1,train_dataset2,padisi_concat_dataset])
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
num_epochs=20
# 定义每个批次中保存的图像数量
num_images_to_save = 3
output_file_path = os.path.join(output_dir, 'image_paths.txt')
with open(output_file_path, 'w') as file:
    for epoch in range(num_epochs):
        for i, (color, depth, ir) in enumerate(train_dataloader):
            # color, depth, ir 分别是加载的三种图像模态数据
            # 每种图像模态进行 PatchMix 操作
            print(color[0].shape)
            # image_c = color[0]  # 假设你想显示批次中的第一张图像
            # image_c_np = image_c.numpy()  # 转换为 numpy 数组
            # image_rgb = cv2.cvtColor(image_c_np, cv2.COLOR_BGR2RGB)
            # plt.imshow(image_rgb)
            # plt.axis('off')  # 关闭坐标轴
            # plt.show()

            # 转换为 NumPy 数组，并在 CPU 上处理
            color_np = color.cpu().numpy()
            print(color_np.shape)
            depth_np = depth.cpu().numpy()
            ir_np = ir.cpu().numpy()

            # 进行图像块的切割和重组，可以分别处理每一种模态
            patch_size = (32, 32)
            stride = 32
            imsize = (96, 96)

            # 对 color 模态进行 PatchMix 操作
            color_im2col = image2cols_batch(image=color_np, patch_size=patch_size, stride=stride)
            color_mask_length = color_im2col.shape[1]
            depth_im2col = image2cols_batch(image=depth_np, patch_size=patch_size, stride=stride)
            depth_mask_length = depth_im2col.shape[1]
            ir_im2col = image2cols_batch(image=ir_np, patch_size=patch_size, stride=stride)
            ir_mask_length = ir_im2col.shape[1]

            color_mixed_patches = np.zeros_like(color_im2col)
            depth_mixed_patches = np.zeros_like(depth_im2col)
            ir_mixed_patches = np.zeros_like(ir_im2col)
            for patch_idx in range(color_mask_length):
                # 随机选择不同图像中的补丁
                # selected_images = np.random.choice(color_im2col.shape[0], color_im2col.shape[0], replace=False)
                # selected_images = np.random.choice(3, 3, replace=False)
                selected_images = np.random.choice(color_im2col.shape[0], 3, replace=False)
                for img_idx, selected_img in enumerate(selected_images):
                    color_mixed_patches[img_idx, patch_idx, :, :, :] = color_im2col[selected_img, patch_idx, :, :, :]
                    depth_mixed_patches[img_idx, patch_idx, :, :, :] = depth_im2col[selected_img, patch_idx, :, :, :]
                    ir_mixed_patches[img_idx, patch_idx, :, :, :] = ir_im2col[selected_img, patch_idx, :, :, :]

                # 将混合后的补丁转换回图像
            color_processed = col2image_batch(coldata=color_mixed_patches, imsize=imsize, stride=stride)
            depth_processed = col2image_batch(coldata=depth_mixed_patches, imsize=imsize, stride=stride)
            ir_processed = col2image_batch(coldata=ir_mixed_patches, imsize=imsize, stride=stride)
            print(color_processed.shape)

            print("color_processed.shape", color_processed.shape)

            for j in range(min(num_images_to_save, color_processed.shape[0])):
                color_image_name = f'color_{epoch}_batch_{i}_{j}.png'
                depth_image_name = f'depth_{epoch}_batch_{i}_{j}.png'
                ir_image_name = f'ir_{epoch}_batch_{i}_{j}.png'

                color_image_path = os.path.join(output_dir, color_image_name)
                depth_image_path = os.path.join(output_dir, depth_image_name)
                ir_image_path = os.path.join(output_dir, ir_image_name)

                # 保存 color 图像
                image_color = color_processed[j]
                image_color_rgb = cv2.cvtColor(image_color.astype(np.uint8), cv2.COLOR_BGR2RGB)
                cv2.imwrite(color_image_path, image_color.astype(np.uint8))

                # 保存 depth 图像
                image_depth = depth_processed[j]
                cv2.imwrite(depth_image_path, image_depth.astype(np.uint8))

                # 保存 ir 图像
                image_ir = ir_processed[j]
                cv2.imwrite(ir_image_path, image_ir.astype(np.uint8))

                # 将图像相对路径写入文件
                file.write(f'{color_image_name} {depth_image_name} {ir_image_name} 0\n')
            # 为了演示，只运行一个 epoch
        # break












