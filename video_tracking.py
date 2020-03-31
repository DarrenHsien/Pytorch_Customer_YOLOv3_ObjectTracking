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
import dlib
from centroidtracker import *
from imutils.video import FPS

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
    print("[INFO] 初始化 argparser 輸入")
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
    #預設幾張辨識一次
    parser.add_argument("--sampling", type=int, default=50, help="image classify frequency")
    opt = parser.parse_args()
    #print("參數檔 : ",opt)
    
    print("[INFO] 設備GPU設定與模型權重加載")
    #=====設定以gpu模式進行運算
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #=====載入模型架構Model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    #=====替模型架構權重載入訓練過得權重
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))
    #=====設定模型目前處理模式為測試模式
    model.eval()


    #=====取得類別標籤
    classes = load_classes(opt.class_path)
    print("[INOF] 當前辨識類別項 : ",classes)
    

    print("[INOF] : 找尋相機...")
    #=====啟動錄像影機
    cap = cv2.VideoCapture(0)
    #=====防呆-無偵測到相機
    assert cap.isOpened(), 'Cannot capture source'
    #=====物件追蹤器
    ct = CentroidTracker(maxDisappeared=40)
    trackers = []
    trackableObjects = {}
    #=====計算FPS
    fps = FPS().start()
    #=====狀態調整用fps
    frames = 0
    #=====去回記數
    totalDown = 0
    totalUp = 0
    
    #取得影像篇幅
    ret, frame = cap.read()
    H,W = frame.shape[:2]
    print("camera image size : H:{} W:{}".format(H,W))
    frame, pad = pad_to_square(frame, 0)
    H,W = frame.shape[:2]
    print("square image size : H:{} W:{}".format(H,W))
    
    assert ret,"無法取得影像"
    print("[INOF] : 開始偵測...")
    while cap.isOpened():
        #=====影像擷取    
        ret, frame = cap.read()
        if ret:
            #=====載入影像輸出方形比例影像
            frame, pad = pad_to_square(frame, 0)
            input_imgs = prep_image(frame,opt.img_size)
            
            #=====系統狀態 : 等待(當系統內無物件追蹤項或非偵測狀態)
            status = "Waiting"
            #=====方框位置儲存清單=====#
            rects = []
            #=====當當前frames滿足sampling->進入AI辨識模式
            if frames % opt.sampling == 0:
                #=====系統狀態 : 偵測(當系統處於AI偵測狀態)
                status = "Detecting"
                #=====追蹤物件清單=====#
                trackers = []
                #=====Pytorch 模組AI偵測
                with torch.no_grad():
                    #=====物件框選輸出
                    detections = model(input_imgs)
                    #=====非最大抑制
                    detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
                #=====當detection偵測到物件時
                if detections[0] is not None:
                    #=====校正 boxes 尺寸至 to original image
                    #=====detection 內容 ： x1,y1,x2,y2,信心度,類別信心度,標籤
                    detections = rescale_boxes(detections[0], opt.img_size, frame.shape[:2])
                    #=====unique_labels : 所有出現在圖像中物件獨立分類編號
                    unique_labels = detections[:, -1].cpu().unique()
                    #=====總共出現總類數
                    n_cls_preds = len(unique_labels)
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                        #Create a Rectangle patch
                        #cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                        if conf > 0.8:
                            #=====採用dlib影像物件追蹤器
                            tracker = dlib.correlation_tracker()
                            #=====建立鎖定方框位置格式
                            rect = dlib.rectangle(x1, y1, x2, y2)
                            #=====追蹤器綁定物件
                            tracker.start_track(frame, rect)
                            #=====追蹤物件清單紀錄追蹤器物件
                            trackers.append(tracker)
            else:
                #=====針對每一個物件追蹤器進行資訊更新(若無追蹤器存在則不會進入loop)
                for tracker in trackers:
                    #=====系統狀態 :追蹤(當系統處於更新追蹤器狀態)
                    status = "Tracking"
                    #=====每個追蹤器須更新物件位置方法則是餵入當下影像
                    tracker.update(frame)
                    #=====取出更新後物件存在位置
                    pos = tracker.get_position()
                    #=====採集x1,x2,y1,y2
                    x1 = int(pos.left())
                    y1 = int(pos.top())
                    x2 = int(pos.right())
                    y2 = int(pos.bottom())
                    #=====方框位置儲存清單增添資訊=====#
                    rects.append((x1, y1, x2, y2))
            
            #=====繪製中心線(用來紀錄人流狀況)
            cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
            #ct為一個用來判定追蹤器所追蹤之物件不衝突篩選器
            objects = ct.update(rects)

            #=====掃描所有追蹤物件->建立物件動向系統
            #=====取出物件編號與物件方框中心位置
            print("trackableObjects : ",trackableObjects) 
            for (objectID, centroid) in objects.items():
                #=====先檢查物件編號是否已存在於trackableObject字典
                to = trackableObjects.get(objectID, None)
                #=====如果該物件編號無在追蹤字典中,則定義的一組
                if to is None:
                    to = TrackableObject(objectID, centroid)
                #=====如果已存在
                else:
                    #=====取得該物件中心移動過路徑之所有y值陣列
                    y = [c[1] for c in to.centroids]
                    #=====取得物件移動方向 : 取得物件中心當前y值-物件平均移動路徑y值
                    direction = centroid[1] - np.mean(y)
                    #=====增添當前路徑
                    to.centroids.append(centroid)
                    #=====如果物件並未被計算過
                    if not to.counted:
                        #=====如果物件移動的方向為向上,且中心點已超過中心線
                        if direction < 0 and centroid[1] < H // 4:
                            totalUp += 1
                            to.counted = True
                        #=====如果物件移動的方向為向下,且中心點已超過中心線
                        elif direction > 0 and centroid[1] > H // 4*3:
                            totalDown += 1
                            to.counted = True
                #=====更新物件動向系統
                trackableObjects[objectID] = to

                # draw both the ID of the object and the centroid of the
                # object on the output frame
                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            # construct a tuple of information we will be displaying on the
            # frame
            info = [
                ("Up", totalUp),
                ("Down", totalDown),
                ("Status", status),
            ]
            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(10) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
            # increment the total number of frames processed thus far and
            # then update the FPS counter
            fps.update()
            frames+=1
            
        # stop the timer and display FPS information
        fps.stop()
        #print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        #print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        


