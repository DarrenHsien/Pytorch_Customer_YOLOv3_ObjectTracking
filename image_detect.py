from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import cv2

def prep_image(img, inp_dim):
    orig_im = img.copy()
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    img_ = img_.cuda()
    return img_


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #使用的yolov3架構
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    #使用的模型權重
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    #使用的模型分類名稱
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    #使用的辨識機率閾值
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    #使用的NMS閾值
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    #預設使用影像尺寸
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    #讀取File路徑
    parser.add_argument("--img_file", type=str, default=None, help="root of each image location")
    opt = parser.parse_args()
    print("設定參數檔",opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device name : ",device)

    # 設定Darknet 53 Model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    # 設定權重讀取項目
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    # 設定模型目前處理模式為測試模式
    model.eval()

    # 擷取類別標籤
    classes = load_classes(opt.class_path)
    print("classes label : ",classes)

    
    print("\nStarting object detection:")
    
    
    while not os.path.isfile(opt.img_file):
        #ignore if no such file is present.
        print("No file exit : ",opt.img_file)
        pass
    
    #讀取影像
    frame = cv2.imread(opt.img_file)
    frame, pad = pad_to_square(frame, 0)
    input_imgs = prep_image(frame,opt.img_size)
    print("Fit in Model img size:",input_imgs.size())
    # 物件偵測
    with torch.no_grad():
        # 物件框選輸出
        detections = model(input_imgs)
        # 非最大抑制
        detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
    # 繪製框架
    # 當detection偵測到兩個以上物件時？
    if detections[0] is not None:
        print(detections)
        # 校正 boxes 尺寸至 to original image
        # detection 內容 ： x1,y1,x2,y2,信心度,類別信心度,標籤
        detections = rescale_boxes(detections[0], opt.img_size, frame.shape[:2])
        # unique_labels : 所有出現在圖像中物件獨立分類編號
        unique_labels = detections[:, -1].cpu().unique()
        # 總共出現總類數
        n_cls_preds = len(unique_labels)
        #bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

            print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

            box_w = x2 - x1
            box_h = y2 - y1

            #color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # Create a Rectangle patch
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
    cv2.imshow("Label",frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
        