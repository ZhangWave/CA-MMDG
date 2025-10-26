import torch

def Real_AdLoss(discriminator_out, criterion, shape_list):
    """
    计算真实样本的对抗损失

    参数:
    discriminator_out (Tensor): 判别器的输出
    criterion (loss function): 损失函数
    shape_list (list): 包含各类别样本数量的列表

    返回:
    Tensor: 真实样本的对抗损失
    """
    print("Discriminator output shape:", discriminator_out.shape)  # 应该是 [batch_size, num_classes]
    
    # 生成对抗标签
    ad_label1_index = torch.LongTensor(shape_list[0], 1).fill_(0)  # 类别0的标签
    ad_label1 = ad_label1_index.cuda()  # 移动到GPU
    ad_label2_index = torch.LongTensor(shape_list[1], 1).fill_(1)  # 类别1的标签
    ad_label2 = ad_label2_index.cuda()  # 移动到GPU
    ad_label3_index = torch.LongTensor(shape_list[2], 1).fill_(2)  # 类别2的标签
    ad_label3 = ad_label3_index.cuda()  # 移动到GPU
    ad_label = torch.cat([ad_label1, ad_label2,ad_label3], dim=0).view(-1)  # 拼接并展平标签
    # print("Ad label shape:", ad_label.shape)  # 应该是 [batch_size]
    real_adloss = criterion(discriminator_out, ad_label)  # 计算真实样本的对抗损失
    return real_adloss

def Fake_AdLoss(discriminator_out, criterion, shape_list):
    """
    计算伪造样本的对抗损失

    参数:
    discriminator_out (Tensor): 判别器的输出
    criterion (loss function): 损失函数
    shape_list (list): 包含各类别样本数量的列表

    返回:
    Tensor: 伪造样本的对抗损失
    """
    # 生成对抗标签
    ad_label1_index = torch.LongTensor(shape_list[0], 1).fill_(0)  # 类别0的标签
    ad_label1 = ad_label1_index.cuda()  # 移动到GPU
    ad_label2_index = torch.LongTensor(shape_list[1], 1).fill_(1)  # 类别1的标签
    ad_label2 = ad_label2_index.cuda()  # 移动到GPU
    ad_label3_index = torch.LongTensor(shape_list[2], 1).fill_(2)  # 类别2的标签
    ad_label3 = ad_label3_index.cuda()  # 移动到GPU
    ad_label = torch.cat([ad_label1, ad_label2,ad_label3], dim=0).view(-1)  # 拼接并展平标签

    fake_adloss = criterion(discriminator_out, ad_label)  # 计算伪造样本的对抗损失
    return fake_adloss

def AdLoss_Limited(discriminator_out, criterion, shape_list):
    """
    计算受限样本的对抗损失

    参数:
    discriminator_out (Tensor): 判别器的输出
    criterion (loss function): 损失函数
    shape_list (list): 包含各类别样本数量的列表

    返回:
    Tensor: 受限样本的对抗损失
    """
    # 生成对抗标签
    ad_label2_index = torch.LongTensor(shape_list[0], 1).fill_(0)  # 类别0的标签
    ad_label2 = ad_label2_index.cuda()  # 移动到GPU
    ad_label3_index = torch.LongTensor(shape_list[1], 1).fill_(1)  # 类别1的标签
    ad_label3 = ad_label3_index.cuda()  # 移动到GPU
    ad_label = torch.cat([ad_label2, ad_label3], dim=0).view(-1)  # 拼接并展平标签

    real_adloss = criterion(discriminator_out, ad_label)  # 计算受限样本的对抗损失
    return real_adloss
