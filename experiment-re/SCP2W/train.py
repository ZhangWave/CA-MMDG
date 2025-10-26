import sys



sys.path.append('../../')
from utils.get_loader_scp2w import get_dataset_scp2w
from utils.utils import save_checkpoint, AverageMeter, Logger, accuracy, mkdirs, adjust_learning_rate, time_to_str
from utils.evaluate import eval
from utils.get_loader import get_dataset
from models.CA_MMDG import DG_model, Discriminator
from loss.hard_triplet_loss import HardTripletLoss
from loss.AdLoss import Real_AdLoss, Fake_AdLoss
import random
import numpy as np
from config import config
from datetime import datetime
import time
from timeit import default_timer as timer

import os
import torch
import torch.nn as nn
import torch.optim as optim 


random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
torch.cuda.manual_seed(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = 'cuda'


def train():
    # 创建必要的目录，包括检查点路径、最佳模型路径和日志目录
    mkdirs(config.checkpoint_path, config.best_model_path, config.logs)

    # 加载数据
    # 调用 get_dataset 函数来加载多个数据集
    src1_train_dataloader_fake, src1_train_dataloader_real, \
        src2_train_dataloader_fake, src2_train_dataloader_real, \
        src3_train_dataloader_fake, src3_train_dataloader_real, \
        tgt_valid_dataloader = get_dataset_scp2w(
        config.src1_data,  # 第一个源数据集的路径或名称
        config.src1_train_num_frames,  # 第一个源数据集中训练数据的帧数
        config.src2_data,  # 第二个源数据集的路径或名称
        config.src2_train_num_frames,  # 第二个源数据集中训练数据的帧数
        config.src3_data,  # 第三个源数据集的路径或名称
        config.src3_train_num_frames,  # 第三个源数据集中训练数据的帧数
        config.tgt_data,  # 目标数据集的路径或名称
        config.tgt_test_num_frames,  # 目标数据集中测试数据的帧数
        config.batch_size  # 批次大小
    )

    # 初始化最佳模型的评估指标
    best_model_ACC = 0.0  # 最佳模型的准确率
    best_model_HTER = 1.0  # 最佳模型的半总错误率
    best_model_ACER = 1.0  # 最佳模型的平均分类错误率
    best_model_AUC = 0.0  # 最佳模型的AUC（曲线下面积）

    # 初始化验证的评估指标列表
    # valid_args 列表依次保存：0:损失, 1:top-1准确率, 2:EER, 3:HTER, 4:ACER, 5:AUC, 6:阈值
    valid_args = [np.inf, 0, 0, 0, 0, 0, 0, 0]

    # 初始化损失和准确率的度量器
    loss_classifier = AverageMeter()  # 分类器的损失度量器
    classifer_top1 = AverageMeter()  # 分类器的top-1准确率度量器

    # 初始化模型和判别器
    net = DG_model(config.model).to(device)  # 领域生成模型
    ad_net_real = Discriminator().to(device)  # 真实样本的判别器
    ad_net_fake = Discriminator().to(device)  # 假样本的判别器

    # 后续代码省略

    # 初始化日志记录器并打开日志文件
    log = Logger()
    log.open(config.logs + config.tgt_data + '_log_SSDG.txt', mode='a')

    # 写入开始时间和一些分隔符到日志文件
    log.write("\n----------------------------------------------- [START %s] %s\n\n" % (
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))

    # 打印和记录正则化标志
    print("Norm_flag: ", config.norm_flag)
    log.write('** start training target model! **\n')

    # 写入日志文件的表头信息
    log.write(
        '--------|------------- VALID -------------|--- classifier ---|------ Current Best ------|--------------|\n')
    log.write(
        '  iter  |   loss   top-1   HTER    AUC    |   loss   top-1   |   top-1   HTER    AUC    |    time      |\n')
    log.write(
        '-------------------------------------------------------------------------------------------------------|\n')

    # 记录当前时间作为计时器的起点
    start = timer()

    # 定义损失函数，包括交叉熵损失和三元组损失
    criterion = {
        'softmax': nn.CrossEntropyLoss().cuda(),
        'triplet': HardTripletLoss(margin=0.1, hardest=False).cuda()
    }

    # 创建优化器的参数字典，包含网络参数和自适应网络参数
    optimizer_dict = [
        {"params": filter(lambda p: p.requires_grad, net.parameters()), "lr": config.init_lr},
        {"params": filter(lambda p: p.requires_grad, ad_net_real.parameters()), "lr": config.init_lr},
    ]

    # 初始化优化器，使用 SGD 优化器
    optimizer = optim.SGD(optimizer_dict, lr=config.init_lr, momentum=config.momentum, weight_decay=config.weight_decay)

    # 初始化参数学习率列表
    init_param_lr = []
    for param_group in optimizer.param_groups:
        init_param_lr.append(param_group["lr"])

    # 每个 epoch 的迭代次数
    iter_per_epoch = 10

    # 初始化真实数据源1的训练数据迭代器和每个 epoch 的迭代次数
    src1_train_iter_real = iter(src1_train_dataloader_real)  # 创建真实数据源1的训练数据迭代器
    src1_iter_per_epoch_real = len(src1_train_iter_real)  # 计算每个 epoch 的真实数据源1的迭代次数

    # 初始化真实数据源2的训练数据迭代器和每个 epoch 的迭代次数
    src2_train_iter_real = iter(src2_train_dataloader_real)  # 创建真实数据源2的训练数据迭代器
    src2_iter_per_epoch_real = len(src2_train_iter_real)  # 计算每个 epoch 的真实数据源2的迭代次数

    # 初始化真实数据源3的训练数据迭代器和每个 epoch 的迭代次数
    src3_train_iter_real = iter(src3_train_dataloader_real)  # 创建真实数据源3的训练数据迭代器
    src3_iter_per_epoch_real = len(src3_train_iter_real)  # 计算每个 epoch 的真实数据源3的迭代次数

    # 初始化伪造数据源1的训练数据迭代器和每个 epoch 的迭代次数
    src1_train_iter_fake = iter(src1_train_dataloader_fake)  # 创建伪造数据源1的训练数据迭代器
    src1_iter_per_epoch_fake = len(src1_train_iter_fake)  # 计算每个 epoch 的伪造数据源1的迭代次数

    # 初始化伪造数据源2的训练数据迭代器和每个 epoch 的迭代次数
    src2_train_iter_fake = iter(src2_train_dataloader_fake)  # 创建伪造数据源2的训练数据迭代器
    src2_iter_per_epoch_fake = len(src2_train_iter_fake)  # 计算每个 epoch 的伪造数据源2的迭代次数

    # 初始化伪造数据源3的训练数据迭代器和每个 epoch 的迭代次数
    src3_train_iter_fake = iter(src3_train_dataloader_fake)  # 创建伪造数据源3的训练数据迭代器
    src3_iter_per_epoch_fake = len(src3_train_iter_fake)  # 计算每个 epoch 的伪造数据源3的迭代次数

    # 最大迭代次数
    max_iter = config.max_iter

    # 初始化 epoch 计数
    epoch = 1

    # 如果使用多个 GPU，使用 DataParallel 包装网络
    if len(config.gpus) > 1:
        net = torch.nn.DataParallel(net).cuda()

    # 迭代训练过程，直到达到最大迭代次数
    # 循环遍历每个迭代次数，包括最大迭代次数 max_iter
    for iter_num in range(max_iter + 1):
        epoch_start = timer()
        torch.cuda.reset_peak_memory_stats()  # 每轮清零统计

        # 如果达到每个源数据集迭代次数的末尾，重新初始化对应的数据迭代器
        if iter_num % src1_iter_per_epoch_real == 0:
            src1_train_iter_real = iter(src1_train_dataloader_real)  # 初始化源数据集1真实数据迭代器
        if iter_num % src2_iter_per_epoch_real == 0:
            src2_train_iter_real = iter(src2_train_dataloader_real)  # 初始化源数据集2真实数据迭代器
        if iter_num % src3_iter_per_epoch_real == 0:
            src3_train_iter_real = iter(src3_train_dataloader_real)  # 初始化源数据集3真实数据迭代器
        if iter_num % src1_iter_per_epoch_fake == 0:
            src1_train_iter_fake = iter(src1_train_dataloader_fake)  # 初始化源数据集1伪造数据迭代器
        if iter_num % src2_iter_per_epoch_fake == 0:
            src2_train_iter_fake = iter(src2_train_dataloader_fake)  # 初始化源数据集2伪造数据迭代器
        if iter_num % src3_iter_per_epoch_fake == 0:
            src3_train_iter_fake = iter(src3_train_dataloader_fake)  # 初始化源数据集3伪造数据迭代器

        # 每次迭代一整轮数据集时，增加 epoch 计数
        if iter_num != 0 and iter_num % iter_per_epoch == 0:
            epoch += 1

        # 保存当前的学习率
        param_lr_tmp = []
        for param_group in optimizer.param_groups:
            param_lr_tmp.append(param_group["lr"])

        # 设置网络和自适应网络为训练模式
        net.train(True)
        ad_net_real.train(True)

        # 清空优化器的梯度
        optimizer.zero_grad()

        # 调整学习率
        adjust_learning_rate(optimizer, epoch, init_param_lr, config.lr_epoch_1, config.lr_epoch_2)

        ######### 数据准备 #########

        # 从源数据集1的真实数据迭代器中获取下一个批次数据，并将其移动到 GPU
        # src1_img_real, src1_label_real = src1_train_iter_real.next()
        src1_img_real, src1_label_real = next(src1_train_iter_real)
        src1_img_real = src1_img_real.cuda()
        src1_label_real = src1_label_real.cuda()
        input1_real_shape = src1_img_real.shape[0]

        # 从源数据集2的真实数据迭代器中获取下一个批次数据，并将其移动到 GPU
        # src2_img_real, src2_label_real = src2_train_iter_real.next()
        src2_img_real, src2_label_real =next(src2_train_iter_real)
        src2_img_real = src2_img_real.cuda()
        src2_label_real = src2_label_real.cuda()
        input2_real_shape = src2_img_real.shape[0]

        # 从源数据集3的真实数据迭代器中获取下一个批次数据，并将其移动到 GPU
        # src3_img_real, src3_label_real = src3_train_iter_real.next()
        src3_img_real, src3_label_real = next(src3_train_iter_real)
        src3_img_real = src3_img_real.cuda()
        src3_label_real = src3_label_real.cuda()
        input3_real_shape = src3_img_real.shape[0]

        # 从源数据集1的伪造数据迭代器中获取下一个批次数据，并将其移动到 GPU
        # src1_img_fake, src1_label_fake = src1_train_iter_fake.next()
        src1_img_fake, src1_label_fake = next(src1_train_iter_fake)
        src1_img_fake = src1_img_fake.cuda()
        src1_label_fake = src1_label_fake.cuda()
        input1_fake_shape = src1_img_fake.shape[0]

        # 从源数据集2的伪造数据迭代器中获取下一个批次数据，并将其移动到 GPU
        # src2_img_fake, src2_label_fake = src2_train_iter_fake.next()
        src2_img_fake, src2_label_fake = next(src2_train_iter_fake)
        src2_img_fake = src2_img_fake.cuda()
        src2_label_fake = src2_label_fake.cuda()
        input2_fake_shape = src2_img_fake.shape[0]

        # 从源数据集3的伪造数据迭代器中获取下一个批次数据，并将其移动到 GPU
        # src3_img_fake, src3_label_fake = src3_train_iter_fake.next()
        src3_img_fake, src3_label_fake = next(src3_train_iter_fake)
        src3_img_fake = src3_img_fake.cuda()
        src3_label_fake = src3_label_fake.cuda()
        input3_fake_shape = src3_img_fake.shape[0]

        # 将所有真实和伪造的数据拼接在一起
        input_data = torch.cat(
            [src1_img_real, src1_img_fake, src2_img_real, src2_img_fake, src3_img_real, src3_img_fake], dim=0)
        source_label = torch.cat(
            [src1_label_real, src1_label_fake, src2_label_real, src2_label_fake, src3_label_real, src3_label_fake],
            dim=0)
        source_label = source_label.view(-1)  # 或 source_label = source_label.squeeze()

        # print("source_label.shape",source_label.shape)

        ######### 前向传播 #########
        classifier_label_out, feature,contrast_loss = net(input_data, config.norm_flag)

        ######### 单边对抗学习 #########
        # 获取各批次数据的特征
        # 计算每个源数据集中真实数据和伪造数据的总数
        input1_shape = input1_real_shape + input1_fake_shape
        input2_shape = input2_real_shape + input2_fake_shape

        # 根据每个源数据集的真实数据和伪造数据的数量，将特征向量 feature 切片成对应的部分
        feature_real_1 = feature.narrow(0, 0, input1_real_shape)  # 切片出源数据集1的真实数据对应的特征
        feature_real_2 = feature.narrow(0, input1_shape, input2_real_shape)  # 切片出源数据集2的真实数据对应的特征
        feature_real_3 = feature.narrow(0, input1_shape + input2_shape, input3_real_shape)  # 切片出源数据集3的真实数据对应的特征
        
        # 根据每个源数据集的伪造数据的数量，将特征向量 feature 切片成对应的部分
        feature_fake_1 = feature.narrow(0, input1_real_shape, input1_fake_shape)  # 切片出源数据集1的伪造数据对应的特征
        feature_fake_2 = feature.narrow(0, input1_shape + input2_real_shape, input2_fake_shape)  # 切片出源数据集2的伪造数据对应的特征
        feature_fake_3 = feature.narrow(0, input1_shape + input2_shape + input3_real_shape,
                                        input3_fake_shape)  # 切片出源数据集3的伪造数据对应的特征

          # 将三个源数据集的伪造数据对应的特征拼接在一起
        feature_fake = torch.cat([feature_fake_1, feature_fake_2, feature_fake_3], dim=0)
        
        # 将三个源数据集的真实数据对应的特征拼接在一起，以便后续的域判别器处理
        feature_real = torch.cat([feature_real_1, feature_real_2, feature_real_3], dim=0)

        # 将拼接后的真实数据对应的特征输入到域判别器进行分类或回归任务
        discriminator_out_real = ad_net_real(feature_real)
        discriminator_out_fake = ad_net_fake(feature_fake)
        ######### 不平衡三元组损失 #########
        # 定义真实和伪造数据的标签
        # 创建真实域标签，标签值分别为0、0、0，表示三个数据集中的真实样本
        real_domain_label_1 = torch.LongTensor(input1_real_shape, 1).fill_(0).cuda()
        real_domain_label_2 = torch.LongTensor(input2_real_shape, 1).fill_(0).cuda()
        real_domain_label_3 = torch.LongTensor(input3_real_shape, 1).fill_(0).cuda()

        # 创建伪造域标签，标签值分别为1、2、3，分别表示三个数据集中的伪造样本
        fake_domain_label_1 = torch.LongTensor(input1_fake_shape, 1).fill_(1).cuda()
        fake_domain_label_2 = torch.LongTensor(input2_fake_shape, 1).fill_(2).cuda()
        fake_domain_label_3 = torch.LongTensor(input3_fake_shape, 1).fill_(3).cuda()

        # 将真实域标签和伪造域标签拼接在一起，形成一个完整的域标签向量
        source_domain_label = torch.cat(
            [real_domain_label_1, fake_domain_label_1, real_domain_label_2, fake_domain_label_2, real_domain_label_3,
             fake_domain_label_3], dim=0).view(-1)

        # 计算三元组损失，使用预定义的损失函数 criterion["triplet"]，并传入生成的特征 feature 和域标签 source_domain_label
        triplet = criterion["triplet"](feature, source_domain_label)

        ######### 交叉熵损失 #########
        real_shape_list = [input1_real_shape, input2_real_shape, input3_real_shape]
        fake_shape_list = [input1_fake_shape, input2_fake_shape, input3_fake_shape]
        real_adloss = 0.5*Real_AdLoss(discriminator_out_real, criterion["softmax"], real_shape_list)+ 0.5*Real_AdLoss(discriminator_out_fake, criterion["softmax"], fake_shape_list)
        cls_loss = criterion["softmax"](classifier_label_out.narrow(0, 0, input_data.size(0)), source_label)

        ######### 反向传播 #########
        # total_loss = cls_loss + config.lambda_triplet * triplet + config.lambda_adreal * real_adloss
        total_loss = cls_loss + config.lambda_adreal * real_adloss
        # print(f"Classification Loss (cls_loss): {cls_loss}")
        # print(f"Triplet Loss (triplet): {triplet} (Weighted by lambda_triplet: {config.lambda_triplet * triplet})")
        # print(f"Real Adversarial Loss (real_adloss): {real_adloss} (Weighted by lambda_adreal: {config.lambda_adreal * real_adloss})")
        # print(f"Contrastive Loss (contrast_loss): {contrast_loss} (Weighted by 0.1: {0.1 * contrast_loss})")
        # print(f"Total Loss (total_loss): {total_loss}")
        

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 更新损失和准确率的度量值
        loss_classifier.update(cls_loss.item())
        acc = accuracy(classifier_label_out.narrow(0, 0, input_data.size(0)), source_label, topk=(1,))
        classifer_top1.update(acc[0])
        # 当前 epoch 耗时
        epoch_duration = timer() - epoch_start
    
    # 到目前为止的平均 epoch 时间
        avg_time_per_epoch = (timer() - start) / (epoch + 1)

        # 打印当前迭代的结果
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # 单位MB
        print('\r', end='', flush=True)
        # print(
        #     '  %4.1f  |  %5.3f  %6.3f  %6.3f  %6.3f  |  %6.3f  %6.3f  |  %6.3f  %6.3f  %6.3f  | %s (%.1f s/epoch)'
        #     % (
        #         (iter_num + 1) / iter_per_epoch,
        #         valid_args[0], valid_args[6], valid_args[3] * 100, valid_args[4] * 100,
        #         loss_classifier.avg, classifer_top1.avg,
        #         float(best_model_ACC), float(best_model_HTER * 100), float(best_model_AUC * 100),
        #         time_to_str(timer() - start, 'min'),
        #         avg_time_per_epoch)
        #     , end='', flush=True)
        print(
            '  %4.1f  |  %5.3f  %6.3f  %6.3f  %6.3f  |  %6.3f  %6.3f  |  %6.3f  %6.3f  %6.3f  | %s (%.1f s/epoch) | PeakMem=%.1fMB'
            % (
                (iter_num + 1) / iter_per_epoch,
                valid_args[0], valid_args[6], valid_args[3] * 100, valid_args[4] * 100,
                loss_classifier.avg, classifer_top1.avg,
                float(best_model_ACC), float(best_model_HTER * 100), float(best_model_AUC * 100),
                time_to_str(timer() - start, 'min'),
                avg_time_per_epoch,
                peak_memory    )
            , end='', flush=True)

        # 每次迭代整轮数据集时进行验证和保存最佳模型
        if iter_num != 0 and (iter_num + 1) % iter_per_epoch == 0:
            # 执行验证，返回验证结果
            # print("tgt_valid_dataloader.shape",tgt_valid_dataloader.shape)
            valid_args = eval(tgt_valid_dataloader, net, config.norm_flag)

            # 根据 HTER 评估是否保存最佳模型
            is_best = valid_args[3] <= best_model_HTER
            best_model_HTER = min(valid_args[3], best_model_HTER)
            threshold = valid_args[5]
            if valid_args[3] <= best_model_HTER:
                best_model_ACC = valid_args[6]
                best_model_AUC = valid_args[4]

            # 保存模型检查点
            save_list = [epoch, valid_args, best_model_HTER, best_model_ACC, best_model_ACER, threshold]
            save_checkpoint(save_list, is_best, net, config.gpus, config.checkpoint_path, config.best_model_path)

            # 打印和记录验证结果
            print('\r', end='', flush=True)
            log.write(
                '  %4.1f  |  %5.3f  %6.3f  %6.3f  %6.3f  |  %6.3f  %6.3f  |  %6.3f  %6.3f  %6.3f  | %s   %s'
                % (
                    (iter_num + 1) / iter_per_epoch,
                    valid_args[0], valid_args[6], valid_args[3] * 100, valid_args[4] * 100,
                    loss_classifier.avg, classifer_top1.avg,
                    float(best_model_ACC), float(best_model_HTER * 100), float(best_model_AUC * 100),
                    time_to_str(timer() - start, 'min'),
                    param_lr_tmp[0]))
            log.write('\n')
            time.sleep(0.01)


if __name__ == '__main__':
    train()




















