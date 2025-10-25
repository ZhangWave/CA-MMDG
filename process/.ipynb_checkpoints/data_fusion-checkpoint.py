import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from process.data_helper import *
from process.augmentation import *
from utils import *

class FDDataset_sufr(Dataset):
    def __init__(self, mode, fold_index = None, sample_type='all',image_size = 128, augment = None, balance = True,
      ):
        super(FDDataset_sufr, self).__init__()
        print('fold: '+str(fold_index))

        self.augment = augment
        self.mode = mode
        self.balance = balance
        self.channels = 3
        self.train_image_path = TRN_IMGS_DIR
        self.test_image_path = TST_IMGS_DIR
        self.image_size = image_size
        self.fold_index = fold_index
        self.sample_type = sample_type  # 新增参数，用于控制输出正负样本
        self.set_mode(self.mode,self.fold_index)



    def set_mode(self, mode, fold_index):
        self.mode = mode
        self.fold_index = fold_index
        print(mode)
        print('fold index set: ', fold_index)

        if self.mode == 'test':
            self.test_list = load_test_list()
            self.num_data = len(self.test_list)
            print('set dataset mode: test')

        elif self.mode == 'val':
            self.val_list = load_val_list()
            self.num_data = len(self.val_list)
            print('set dataset mode: val')

        elif self.mode == 'train':
            self.train_list = load_train_list()
            random.shuffle(self.train_list)
            self.num_data = len(self.train_list)
            print('set dataset mode: train')
            # 调试信息：打印train_list前几个样本
            print('First few samples in train_list:', self.train_list[:5])

 # 分开正负样本
            self.train_list_fake = [x for x in self.train_list if x[3] == '0']
            self.train_list_real = [x for x in self.train_list if x[3] == '1']

            # 调试信息：打印样本数量
            print('Number of fake samples (a):', len(self.train_list_fake))
            print('Number of real samples:', len(self.train_list_real))
            print('Total number of samples (num_data):', self.num_data)

        print(self.num_data)

    def __getitem__(self, index,domain_label=0):
        if self.mode == 'train':
            if self.sample_type == 'fake':
                sample_list = self.train_list_fake
            elif self.sample_type == 'real':
                sample_list = self.train_list_real
            else:
                sample_list = self.train_list

            index = index % len(sample_list)
            color, depth, ir, label = sample_list[index]

        elif self.mode == 'val':
            color, depth, ir, label = self.val_list[index]

        elif self.mode == 'test':
            color,depth,ir = self.test_list[index]
            test_id = color+' '+depth+' '+ir

            # 修改路径设置，以确保验证集使用正确的路径
        if self.mode == 'val':
            data_path = '/phase1/valid'  # 验证集的路径
        # if self.mode == 'val':
        #     data_path = '/phase2/test'  # 验证集的路径
        elif self.mode == 'train':
            data_path = '/phase1/train'  # 训练集和测试集的路径
        elif self.mode == 'test':
            data_path = '/phase2/test'

        # print(color)
        parts = color.split('/')
        ID = '/'.join(parts[1:3])
        # print(ID)

        color = cv2.imread(os.path.join(DATA_ROOT+data_path, color),1)
        depth = cv2.imread(os.path.join(DATA_ROOT+data_path, depth),1)
        ir = cv2.imread(os.path.join(DATA_ROOT+data_path, ir),1)

        color = cv2.resize(color,(RESIZE_SIZE,RESIZE_SIZE))
        depth = cv2.resize(depth,(RESIZE_SIZE,RESIZE_SIZE))
        ir = cv2.resize(ir,(RESIZE_SIZE,RESIZE_SIZE))

        if self.mode == 'train':
            # color = color_augumentor(color,target_shape=(self.image_size, self.image_size, 3))
            # depth = color_augumentor(depth,target_shape=(self.image_size, self.image_size, 3))
            # ir = color_augumentor(ir,target_shape=(self.image_size, self.image_size, 3))

            color = cv2.resize(color, (self.image_size, self.image_size))
            depth = cv2.resize(depth, (self.image_size, self.image_size))
            color = cv2.resize(ir, (self.image_size, self.image_size))

            image = np.concatenate([color.reshape([self.image_size, self.image_size, 3]),
                                    depth.reshape([self.image_size, self.image_size, 3]),
                                    ir.reshape([self.image_size, self.image_size, 3])], axis=2)

            if random.randint(0, 1) == 0:
                random_pos = random.randint(0, 2)
                if random.randint(0, 1) == 0:
                    image[:, :, 3 * random_pos:3 * (random_pos + 1)] = 0
                else:
                    for i in range(3):
                        if i != random_pos:
                            image[:, :, 3 * i:3 * (i + 1)] = 0

            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            image = image.reshape([self.channels * 3, self.image_size, self.image_size])
            image = image / 255.0

            label = int(label)



            return torch.FloatTensor(image), torch.LongTensor(np.asarray(label).reshape([-1]))

        elif self.mode == 'val':
            # 将图像堆叠成一个多通道图像
            image = np.concatenate([color, depth, ir], axis=2)  # 注意这里的axis参数

            # 重新排列轴以符合PyTorch的期望格式 [C, H, W]
            image = np.transpose(image, (2, 0, 1))

            # 将图像数据类型转换为float32，并归一化
            image = image.astype(np.float32) / 255.0

            # 将图像转换为torch tensor
            # image_tensor = torch.FloatTensor(image).unsqueeze(0)  # 添加一个批次维度
            image_tensor = torch.FloatTensor(image)  # 不再添加一个批次维度
            label=int(label)

            # 处理标签
            label_tensor = torch.LongTensor(np.asarray(label).reshape([-1]))

            return image_tensor, label_tensor,ID

        elif self.mode == 'test':

            # 将图像堆叠成一个多通道图像
            image = np.concatenate([color, depth, ir], axis=2)  # 注意这里的axis参数

            # 重新排列轴以符合PyTorch的期望格式 [C, H, W]
            image = np.transpose(image, (2, 0, 1))

            # 将图像数据类型转换为float32，并归一化
            image = image.astype(np.float32) / 255.0

            # 由于处理的是单个图像，所以这里不需要reshape操作
            # 直接将图像转换为torch tensor即可
            image_tensor = torch.FloatTensor(image).unsqueeze(0)  # 添加一个批次维度
            return image_tensor, test_id

    def __len__(self):
        return self.num_data

# check #################################################################
def run_check_train_data():
    dataset = FDDataset_sufr(mode = 'val',image_size=48)
    print(dataset)

    num = 5
    for m in range(num):
        i = np.random.choice(num)
        image, label,ID = dataset[m]
        print(image.shape)
        print(label)
        print(ID)

        if m > 100:
            break

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_check_train_data()


