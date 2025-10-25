import os
import random
from utils import *

# 设置数据根目录
DATA_ROOT =  r'F:\BaiduNetdiskDownload\patchmix'
# 设置训练和测试图像的目录

# 设置图片调整大小后的大小
RESIZE_SIZE = 112

# 加载训练列表
def load_train_list_scp():
    list = []
    f = open(DATA_ROOT + '/scp/z_scp.txt')
    lines = f.readlines()

    for line in lines:
        # 移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
        line = line.strip().split(' ')
        list.append(line)
    return list       #输出形式：'Training/fake_part/CLKJ_AS0005/04_en_b.rssdk/color/101.jpg'



# 加载验证列表
def load_val_list():
    list = []
    f = open(DATA_ROOT + '/phase1/val_private_list.txt')
    lines = f.readlines()

    for line in lines:
        line = line.strip().split(' ')
        list.append(line)
        #['Val/0000/000000-color.jpg', 'Val/0000/000000-depth.jpg', 'Val/0000/000000-ir.jpg', '0']
    return list

# 加载测试列表
def load_test_list():
    list = []
    f = open(DATA_ROOT + '/phase2/test/test_public_list.txt')
    lines = f.readlines()

    for line in lines:
        line = line.strip().split(' ')
        list.append(line)

    return list

# 对训练列表进行平衡处理
def transform_balance(train_list):
    pos_list = []
    neg_list = []
    for tmp in train_list:
        # 根据标签将数据分为正负两类
        if tmp[3]=='1':
            pos_list.append(tmp)
        else:
            neg_list.append(tmp)

    # 打印正负样本数量
    print(len(pos_list))
    print(len(neg_list))
    return [pos_list,neg_list]

def submission(probs, outname, mode='valid'):
    # 根据提供的模式（验证或测试），选择相应的列表文件
    # 这个列表文件包含了数据样本的标识符
    if mode == 'valid':
        f = open(DATA_ROOT + '/phase1/valid/val_public_list.txt')
    else:
        f = open(DATA_ROOT + '/phase2/test/test_public_list.txt')

    # 读取列表文件的所有行，并关闭文件
    lines = f.readlines()
    f.close()
    # 移除每行文本末尾的换行符和空白字符
    lines = [tmp.strip() for tmp in lines]

    # 打开输出文件，准备写入预测结果
    f = open(outname, 'w')
    # 使用zip函数将样本标识符与对应的概率配对
    for line, prob in zip(lines, probs):
        # 创建一个包含样本标识符和概率的字符串
        out = line + ' ' + str(prob)
        # 将字符串写入文件，并在每个预测后添加换行符
        f.write(out + '\n')
    # 关闭输出文件
    f.close()
    # 函数返回一个名为 'list' 的变量，但在这段代码中似乎并未定义，
    # 可能是一个错误或遗漏，需要进一步的确认
    return list

# 主函数
if __name__ == '__main__':
    # 加载测试列表

    # load_val_list=load_val_list()
    load_train_list=load_train_list_scp()
    # load_test_list=load_test_list()


    # 打印前10项
    for item in load_train_list[:10]:
        print(item)
    print('***********************************')


