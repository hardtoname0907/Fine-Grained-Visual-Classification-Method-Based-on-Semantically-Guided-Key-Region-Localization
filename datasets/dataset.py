import numpy as np
import imageio
import os
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class CUB():
    def __init__(self, input_size, root, is_train=True, data_len=None):
        self.input_size = input_size
        self.root = root
        self.is_train = is_train
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        box_file = open(os.path.join(self.root, 'bounding_boxes.txt'))
        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))
        box_file_list = []
        for line in box_file:
            data = line[:-1].split(' ')
            box_file_list.append([int(float(data[2])), int(float(data[1])),
                                  int(float(data[4])), int(float(data[3]))])
        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
        self.train_box = torch.tensor([x for i, x in zip(train_test_list, box_file_list) if i])
        self.test_box = torch.tensor([x for i, x in zip(train_test_list, box_file_list) if not i])
        if self.is_train:
            self.train_img = [os.path.join(self.root, 'images', train_file) for train_file in
                              train_file_list[:data_len]]
            # self.train_img = [imageio.imread(os.path.join(self.root, 'images', train_file)) for train_file in
            #                   train_file_list[:data_len]]
            self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
        if not self.is_train:
            self.test_img = [os.path.join(self.root, 'images', test_file) for test_file in
                             test_file_list[:data_len]]
            # self.test_img = [imageio.imread(os.path.join(self.root, 'images', test_file)) for test_file in
            #                  test_file_list[:data_len]]
            self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]

    def __getitem__(self, index):
        if self.is_train:
            img, target, box = imageio.imread(self.train_img[index]), self.train_label[index], self.train_box[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')

            # compute scaling
            height, width = img.height, img.width
            height_scale = self.input_size / height
            width_scale = self.input_size / width

            img = transforms.Resize((self.input_size, self.input_size), Image.BILINEAR)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)

            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target, box = imageio.imread(self.test_img[index]), self.test_label[index], self.test_box[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')

            # compute scaling
            height, width = img.height, img.width
            height_scale = self.input_size / height
            width_scale = self.input_size / width

            img = transforms.Resize((self.input_size, self.input_size), Image.BILINEAR)(img)
            # img = transforms.Resize((688, 688), Image.BILINEAR)(img)
            # img = transforms.CenterCrop(448)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        scale = torch.tensor([height_scale, width_scale])

        return img, target, box, scale

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)

class STANFORD_CAR():
    def __init__(self, input_size, root, is_train=True, data_len=None):
        self.input_size = input_size
        self.root = root
        self.is_train = is_train
        train_img_path = os.path.join(self.root, 'cars_train')
        test_img_path = os.path.join(self.root, 'cars_test')
        train_label_file = open(os.path.join(self.root, 'train.txt'))
        test_label_file = open(os.path.join(self.root, 'test.txt'))
        train_img_label = []
        test_img_label = []
        for line in train_label_file:
            train_img_label.append([os.path.join(train_img_path,line[:-1].split(' ')[0]), int(line[:-1].split(' ')[1])-1])
        for line in test_label_file:
            test_img_label.append([os.path.join(test_img_path,line[:-1].split(' ')[0]), int(line[:-1].split(' ')[1])-1])
        self.train_img_label = train_img_label[:data_len]
        self.test_img_label = test_img_label[:data_len]


    def __getitem__(self, index):
        if self.is_train:
            img, target = imageio.imread(self.train_img_label[index][0]), self.train_img_label[index][1]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')

            img = transforms.Resize((self.input_size, self.input_size), Image.BILINEAR)(img)
            # img = transforms.RandomResizedCrop(size=self.input_size,scale=(0.4, 0.75),ratio=(0.5,1.5))(img)
            # img = transforms.RandomCrop(self.input_size)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)

            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target = imageio.imread(self.test_img_label[index][0]), self.test_img_label[index][1]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((self.input_size, self.input_size), Image.BILINEAR)(img)
            # img = transforms.CenterCrop(self.input_size)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)


        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_img_label)
        else:
            return len(self.test_img_label)

class FGVC_aircraft():
    def __init__(self, input_size, root, is_train=True, data_len=None):
        self.input_size = input_size
        self.root = root
        self.is_train = is_train
        train_img_path = os.path.join(self.root, 'data', 'images')
        test_img_path = os.path.join(self.root, 'data', 'images')
        train_label_file = open(os.path.join(self.root, 'data', 'train.txt'))
        test_label_file = open(os.path.join(self.root, 'data', 'test.txt'))
        train_img_label = []
        test_img_label = []
        for line in train_label_file:
            train_img_label.append([os.path.join(train_img_path,line[:-1].split(' ')[0]), int(line[:-1].split(' ')[1])-1])
        for line in test_label_file:
            test_img_label.append([os.path.join(test_img_path,line[:-1].split(' ')[0]), int(line[:-1].split(' ')[1])-1])
        self.train_img_label = train_img_label[:data_len]
        self.test_img_label = test_img_label[:data_len]

    def __getitem__(self, index):
        if self.is_train:
            img, target = imageio.imread(self.train_img_label[index][0]), self.train_img_label[index][1]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')

            img = transforms.Resize((self.input_size, self.input_size), Image.BILINEAR)(img)
            # img = transforms.RandomResizedCrop(size=self.input_size,scale=(0.4, 0.75),ratio=(0.5,1.5))(img)
            # img = transforms.RandomCrop(self.input_size)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)

            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target = imageio.imread(self.test_img_label[index][0]), self.test_img_label[index][1]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((self.input_size, self.input_size), Image.BILINEAR)(img)
            # img = transforms.CenterCrop(self.input_size)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_img_label)
        else:
            return len(self.test_img_label)


class STANFORD_DOG():
    def __init__(self, input_size, root, is_train=True, data_len=None):
        self.input_size = input_size
        self.root = root
        self.imgroot='/data/public/Standford_Dogs'
        self.is_train = is_train
        train_img_path = os.path.join(self.imgroot, 'Images')
        test_img_path = os.path.join(self.imgroot, 'Images')
        train_label_file = open(os.path.join(self.root, 'train.txt'))
        test_label_file = open(os.path.join(self.root, 'test.txt'))
        train_img_label = []
        test_img_label = []
        for line in train_label_file:
            train_img_label.append([os.path.join(train_img_path,line[:-1].split(' ')[0]), int(line[:-1].split(' ')[1])-1])
        for line in test_label_file:
            test_img_label.append([os.path.join(test_img_path,line[:-1].split(' ')[0]), int(line[:-1].split(' ')[1])-1])
        self.train_img_label = train_img_label[:data_len]
        self.test_img_label = test_img_label[:data_len]

    def __getitem__(self, index):
        if self.is_train:
            img, target = imageio.imread(self.train_img_label[index][0]), self.train_img_label[index][1]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')

            img = transforms.Resize((self.input_size, self.input_size), Image.BILINEAR)(img)
            # img = transforms.RandomResizedCrop(size=self.input_size,scale=(0.4, 0.75),ratio=(0.5,1.5))(img)
            # img = transforms.RandomCrop(self.input_size)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)

            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target = imageio.imread(self.test_img_label[index][0]), self.test_img_label[index][1]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((self.input_size, self.input_size), Image.BILINEAR)(img)
            # img = transforms.CenterCrop(self.input_size)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_img_label)
        else:
            return len(self.test_img_label)

class NAbirds():
    def __init__(self, input_size, root, is_train=True, data_len=None):
        self.input_size = input_size
        self.imgroot = '/data/public/nabirds'
        self.root = root
        self.is_train = is_train
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        box_file = open(os.path.join(self.root, 'bounding_boxes.txt'))
        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))
        box_file_list = []
        for line in box_file:
            data = line[:-1].split(' ')
            box_file_list.append([int(float(data[2])), int(float(data[1])),
                                  int(float(data[4])), int(float(data[3]))])
        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
        self.train_box = torch.tensor([x for i, x in zip(train_test_list, box_file_list) if i])
        self.test_box = torch.tensor([x for i, x in zip(train_test_list, box_file_list) if not i])

        # 创建类别映射
        self.class_labels = sorted(set(label_list))
        self.class_map = {label: idx for idx, label in enumerate(self.class_labels)}
    
        if self.is_train:
            self.train_img = [os.path.join(self.imgroot, 'images', train_file) for train_file in
                              train_file_list[:data_len]]
            # self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
            self.train_label = [self.class_map[x] for i, x in zip(train_test_list, label_list) if i][:data_len]
        if not self.is_train:
            self.test_img = [os.path.join(self.imgroot, 'images', test_file) for test_file in
                             test_file_list[:data_len]]
            # self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]
            self.test_label = [self.class_map[x] for i, x in zip(train_test_list, label_list) if not i][:data_len]

    def __getitem__(self, index):
        if self.is_train:
            img, target, box = imageio.imread(self.train_img[index]), self.train_label[index], self.train_box[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')

            # compute scaling
            height, width = img.height, img.width
            height_scale = self.input_size / height
            width_scale = self.input_size / width

            img = transforms.Resize((self.input_size, self.input_size), Image.BILINEAR)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)

            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target, box = imageio.imread(self.test_img[index]), self.test_label[index], self.test_box[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')

            # compute scaling
            height, width = img.height, img.width
            height_scale = self.input_size / height
            width_scale = self.input_size / width

            img = transforms.Resize((self.input_size, self.input_size), Image.BILINEAR)(img)
            # img = transforms.Resize((688, 688), Image.BILINEAR)(img)
            # img = transforms.CenterCrop(448)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        scale = torch.tensor([height_scale, width_scale])

        return img, target #, box, scale

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)


class Luxury():
    def __init__(self, input_size, root, is_train=True, data_len=None):
        self.input_size = input_size
        self.root = root
        self.is_train = is_train
        self.data_len = data_len

        # 定义训练和测试目录
        self.train_dir = os.path.join(self.root, 'train')
        self.test_dir = os.path.join(self.root, 'test')

        # 定义数据增强和预处理
        self.train_transforms = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size), Image.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.test_transforms = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # 使用 ImageFolder 加载数据集
        if self.is_train:
            self.dataset = datasets.ImageFolder(root=self.train_dir, transform=self.train_transforms)
        else:
            self.dataset = datasets.ImageFolder(root=self.test_dir, transform=self.test_transforms)

        # 如果需要限制数据集长度
        # if self.data_len is not None:
        #     self.dataset = torch.utils.data.Subset(self.dataset, list(range(self.data_len)))

    def __getitem__(self, index):
        img, target = self.dataset[index]
        return img, target

    def __len__(self):
        return len(self.dataset)

class Geners():
    def __init__(self, input_size, root, is_train=True, data_len=None):
        self.input_size = input_size
        self.root = root
        self.is_train = is_train
        self.data_len = data_len

        # 定义训练和测试目录
        self.train_dir = os.path.join(self.root, 'train')
        self.test_dir = os.path.join(self.root, 'test')

        # 定义数据增强和预处理
        self.train_transforms = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size), Image.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.test_transforms = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # 使用 ImageFolder 加载数据集
        if self.is_train:
            self.dataset = datasets.ImageFolder(root=self.train_dir, transform=self.train_transforms)
        else:
            self.dataset = datasets.ImageFolder(root=self.test_dir, transform=self.test_transforms)

        # 如果需要限制数据集长度
        # if self.data_len is not None:
        #     self.dataset = torch.utils.data.Subset(self.dataset, list(range(self.data_len)))

    def __getitem__(self, index):
        img, target = self.dataset[index]
        return img, target

    def __len__(self):
        return len(self.dataset)
