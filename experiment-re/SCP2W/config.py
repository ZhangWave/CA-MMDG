class DefaultConfigs(object):
    # 设置随机种子，保证结果的可重复性
    seed = 666

    # SGD优化器的参数
    weight_decay = 5e-4  # 权重衰减，防止过拟合
    momentum = 0.9  # 动量，加速收敛

    # 学习率设置
    init_lr = 0.01  # 初始学习率
    lr_epoch_1 = 0  # 学习率调整的第一个epoch
    lr_epoch_2 = 150  # 学习率调整的第二个epoch   150

    # 模型设置
    pretrained = True  # 是否使用预训练模型
    model = 'resnet18'  # 使用的模型，可以是'resnet18'或'maddg'

    # 训练参数
    gpus = "0"  # 使用的GPU编号
    batch_size =32   # 批量大小
    norm_flag = True  # 是否进行数据归一化
    max_iter = 4000  # 最大迭代次数
    lambda_triplet = 2  # 三元组损失的权重
    lambda_adreal = 0.1  # Adversarial Realism损失的权重

    # 测试模型的名称
    tgt_best_model_name = 'model_best_0.13538_53.pth.tar'

    # 目标数据集信息
    tgt_data = 'wmca'  # 目标数据集

    # 路径信息
    checkpoint_path = './' + tgt_data + '_checkpoint/' + model + '/DGFANet/'  # 检查点保存路径
    best_model_path = './' + tgt_data + '_checkpoint/' + model + '/best_model/'  # 最佳模型保存路径
    logs = './logs/'  # 日志保存路径


# 实例化配置对象
config = DefaultConfigs()
