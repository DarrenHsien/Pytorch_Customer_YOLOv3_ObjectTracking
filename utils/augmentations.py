import torch
import torch.nn.functional as F
import numpy as np
import albumentations as A


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets

# images[np.unit8], boxes[numpy] = (cls, x, y, w, h)
def augmentMix(image, boxes):
    print("Input boxes : ",boxes)
    _,h, w = image.shape
    print("image size : ",h,w)
    labels, boxes_coord = boxes[:, 1], boxes[:, 2:]
    labels = labels.tolist()
    boxes_coord = boxes_coord * h     # 得到原图尺寸下的坐标（未归一化的坐标）
    boxes_coord[:, 0] = np.clip(boxes_coord[:, 0]-boxes_coord[:, 2]/2, a_min=0, a_max=None)   # 确保x_min和y_min有效
    boxes_coord[:, 1] = np.clip(boxes_coord[:, 1]-boxes_coord[:, 3]/2, a_min=0, a_max=None)
    boxes_coord = boxes_coord.tolist()      # [x_min, y_min, width, height]
    print("Input boxes_coord : ",boxes_coord)
    print("Input labels : ",labels)
    # 在这里设置数据增强的方法
    aug = A.Compose([
        # 水平翻轉
        A.HorizontalFlip(p=0.5),
        # 飽和度差異
        #A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.5),
        # 垂直翻轉
        #A.VerticalFlip(p=0.5),
        # 模糊
        #A.Blur(p=1),
        # 隨意縮放裁切
        #A.RandomResizedCrop(h, w, scale=(0.7, 1.5), ratio=(0.75, 1.2), p=0.5),
#       # 平移旋轉
        #A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=5, border_mode=0, p=0.5)
    ], bbox_params={'format':'coco', 'label_fields': ['category_id']})
    augmented = aug(image=image, bboxes=boxes_coord, category_id=labels)
    # 经过aug之后，如果把boxes变没了，则返回原来的图片
    if augmented['bboxes']:
        image = augmented['image']

        boxes_coord = np.array(augmented['bboxes']) # x_min, y_min, w, h → x, y, w, h
        boxes_coord[:, 0] = boxes_coord[:, 0] + boxes_coord[:, 2]/2
        boxes_coord[:, 1] = boxes_coord[:, 1] + boxes_coord[:, 3]/2
        boxes_coord = boxes_coord / h
        labels = np.array(augmented['category_id'])[:, None]
        print("labels.shape : ",labels.shape)
        boxes = np.concatenate((np.array([0]),labels), 1)
        print("boxes : ",boxes)
        boxes = np.concatenate((boxes, boxes_coord), 1)
    print("Output boxes : ",boxes)
    return image, boxes



# images[np.unit8], boxes[numpy] = (cls, x, y, w, h)
def augment(image, boxes):
    #print("Input boxes : ",boxes)
    h, w, _= image.shape
    #print("image size : ",h,w)
    labels, boxes_coord = boxes[:, 0], boxes[:, 1:]
    labels = labels.tolist()
    #print("Input boxes_coord0 : ",boxes_coord)
    boxes_coord = boxes_coord * h     # 得到原图尺寸下的坐标（未归一化的坐标）
    #print("Input boxes_coord1 : ",boxes_coord)
    boxes_coord[:, 0] = np.clip(boxes_coord[:, 0]-boxes_coord[:, 2]/2, a_min=0, a_max=None)   # 确保x_min和y_min有效
    boxes_coord[:, 1] = np.clip(boxes_coord[:, 1]-boxes_coord[:, 3]/2, a_min=0, a_max=None)
    #print("Input boxes_coord2 : ",boxes_coord)
    for idx, aboxes_coord in enumerate(boxes_coord):
        if aboxes_coord[2]+aboxes_coord[0] > w:
            boxes_coord[idx][2]=w-aboxes_coord[0]
        if aboxes_coord[3]+aboxes_coord[1] > h:
            boxes_coord[idx][3]=h-aboxes_coord[1]
    
    boxes_coord = boxes_coord.tolist()      # [x_min, y_min, width, height]
    #print("Input boxes_coord3 : ",boxes_coord)
    #print("Input labels : ",labels)
    # 在这里设置数据增强的方法
    aug = A.Compose([
        # 水平翻轉
        A.HorizontalFlip(p=0.5),
        # 飽和度差異
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.5),
        # 垂直翻轉
        #A.VerticalFlip(p=0.5),
        # 模糊
        A.Blur(p=0.5),
        # 隨意縮放裁切
        A.RandomResizedCrop(h, w, scale=(0.7, 1.5), ratio=(0.75, 1.2), p=0.5),
#       # 平移旋轉
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=5, border_mode=0, p=0.5)
    ], bbox_params={'format':'coco', 'label_fields': ['category_id']})

    augmented = aug(image=image, bboxes=boxes_coord, category_id=labels)

    # 经过aug之后，如果把boxes变没了，则返回原来的图片
    if augmented['bboxes']:
        image = augmented['image']

        boxes_coord = np.array(augmented['bboxes']) # x_min, y_min, w, h → x, y, w, h
        boxes_coord[:, 0] = boxes_coord[:, 0] + boxes_coord[:, 2]/2
        boxes_coord[:, 1] = boxes_coord[:, 1] + boxes_coord[:, 3]/2
        boxes_coord = boxes_coord / h
        #print("boxes_coord : ",boxes_coord)
        labels = np.array(augmented['category_id'])[:, None]
        boxes = np.concatenate((labels, boxes_coord), 1)

    return image, boxes