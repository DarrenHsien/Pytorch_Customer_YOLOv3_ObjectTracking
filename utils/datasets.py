import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils.augmentations import horisontal_flip,augmentMix,augment
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# 填充影像篇幅左右上下
# 对numpy格式的img进行padding([0,255])
def pad_to_square(img, pad_value):
    h, w, _ = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = ((pad1, pad2), (0,0), (0,0)) if h <= w else ((0,0), (pad1, pad2), (0,0))  # 分别对应h,w,c的padding
    # Add padding
    img = np.pad(img, pad, 'constant', constant_values=pad_value)

    return img, (*pad[1], *pad[0])  # 返回w，c的padding



def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        #讀取train.txt
        with open(list_path, "r") as file:
            self.img_files = file.readlines()
        #整合影像與標籤
        #self.img_files
        #self.label_files
        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        #定義輸入影像尺寸大小
        self.img_size = img_size
        #單張影像最大允許物件數
        self.max_objects = 100
        #是否使用影像擴增技術
        self.augment = augment
        #是否搭配影像隨機縮放比例(self.min_size~self.max_size)
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        #是否將標籤原本以百分比標記還原為影像對應尺寸
        self.normalized_labels = normalized_labels
        #紀錄batch次數
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        #擷取影像路徑
        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # 將影像轉換為tensor
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)

        # 擴增影像層數滿足3層
        if len(img.shape) != 3:
            img = img[None, :, :]
            img = img.repeat(3, 0)

        #是否轉換標籤百分比為實際影像尺寸量度
        h, w, _ = img.shape  # np格式的img是H*W*C
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # 填補影像至正方形尺寸
        img, pad = pad_to_square(img, 0)
        padded_h, padded_w, _ = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()
        assert os.path.exists(label_path)   # 确保label_path必定存在，即图片必定存在label
        

        boxes = np.loadtxt(label_path).reshape(-1, 5)
        # Extract coordinates for unpadded + unscaled image
        x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
        y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
        x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
        y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
        # Adjust for added padding
        x1 += pad[0]    # pad是从低维到高维的，感觉这样写是有问题的，应该只与pad[0][2]有关，不过一般都是相等的
        y1 += pad[2]
        x2 += pad[0]
        y2 += pad[2]
        # Returns (x, y, w, h)
        boxes[:, 1] = ((x1 + x2) / 2) / padded_w
        boxes[:, 2] = ((y1 + y2) / 2) / padded_h
        boxes[:, 3] *= w_factor / padded_w      # 原来的数值是boxw_ori/imgw_ori, 现在变成了(boxw_ori/imgw_ori)*imgw_ori/imgw_pad=boxw_ori/imgw_pad
        boxes[:, 4] *= h_factor / padded_h

        # Apply augmentations
        # img, 以最长边为标准进行padding得到的uint8图像
        # boxes, (cls, x, y, w, h)都以pad后得到的img的高度进行了归一化
        if self.augment:
            img, boxes = augment(img, boxes)

        img = transforms.ToTensor()(img)   # ToTensor已经将像素值进行了归一化

        targets = torch.zeros((len(boxes), 6))
        targets[:, 1:] = torch.from_numpy(boxes)  # 0维在collate_fn中是作为idx用了，用于指定target对应的图片

        return img_path, img, targets


    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)
