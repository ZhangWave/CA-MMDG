import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from process.data_helper_padisi import *
from process.augmentation_padisi import *
from utils import *

class FDDataset_padisi(Dataset):
    def __init__(self, mode,sample_type='all' ,fold_index = None, image_size = 128, augment = None, balance = True,
      ):
        super(FDDataset_padisi, self).__init__()
        print('fold: '+str(fold_index))

        self.augment = augment
        self.mode = mode
        self.balance = balance
        self.channels = 3
        self.train_image_path = TRN_IMGS_DIR
        self.test_image_path = TST_IMGS_DIR
        self.image_size = image_size
        self.fold_index = fold_index
        self.set_mode(self.mode,self.fold_index)
        self.sample_type = sample_type  # 新增参数，用于控制输出正负样本



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

            # if self.balance:
            #     self.train_list = transform_balance(self.train_list)
            # 分开正负样本
            self.train_list_fake = [x for x in self.train_list if x[3] == '0']
            self.train_list_real = [x for x in self.train_list if x[3] == '1']

        print(self.num_data)

    def __getitem__(self, index,domain_label=2):
        if self.mode == 'train':
            if self.sample_type == 'fake':
                sample_list = self.train_list_fake
            elif self.sample_type == 'real':
                sample_list = self.train_list_real
            else:
                sample_list = self.train_list

            index = index % len(sample_list)
            # print("index",index)

            color, depth, ir, label = sample_list[index]

        elif self.mode == 'val':
            color, depth, ir, label = self.val_list[index]

        elif self.mode == 'test':
            # color,depth,ir = self.test_list[index]
            color, depth, ir, *rest = self.test_list[index]
            test_id = color+' '+depth+' '+ir
            # color, depth, ir, *rest = self.test_list[index]

        # print(color)
        parts = color.split('/')
        ID = parts[1].split('_')[1]

        # print(ID)

        color = cv2.imread(os.path.join(DATA_ROOT, color),1)
        depth = cv2.imread(os.path.join(DATA_ROOT, depth),1)
        ir = cv2.imread(os.path.join(DATA_ROOT, ir),1)

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


            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            image = image.reshape([self.channels * 3, self.image_size, self.image_size])
            image = image / 255.0
            # print("image.shape",image.shape)

            label = int(label)
            # a = torch.LongTensor([label])  # 将label转为张量
            # print("label",a.shape)



            return torch.FloatTensor(image), torch.LongTensor(np.asarray(label).reshape([-1]))

        elif self.mode == 'val':

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
            color = color_augumentor(color, target_shape=(self.image_size, self.image_size, 3), is_infer=True)
            depth = depth_augumentor(depth, target_shape=(self.image_size, self.image_size, 3), is_infer=True)
            ir = ir_augumentor(ir, target_shape=(self.image_size, self.image_size, 3), is_infer=True)
            n = len(color)

            color = np.concatenate(color, axis=0)
            depth = np.concatenate(depth, axis=0)
            ir = np.concatenate(ir, axis=0)

            image = np.concatenate([color.reshape([n, self.image_size, self.image_size, 3]),
                                    depth.reshape([n, self.image_size, self.image_size, 3]),
                                    ir.reshape([n, self.image_size, self.image_size, 3])],
                                   axis=3)

            image = np.transpose(image, (0, 3, 1, 2))
            image = image.astype(np.float32)
            image = image.reshape([n, self.channels * 3, self.image_size, self.image_size])
            image = image / 255.0

            return torch.FloatTensor(image), test_id

    def __len__(self):

        return self.num_data

# check #################################################################
def run_check_train_data():
    dataset = FDDataset_padisi(mode = 'val')
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


