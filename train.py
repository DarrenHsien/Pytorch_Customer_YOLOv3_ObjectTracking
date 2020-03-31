from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 訓練步伐
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    # 一次導入幾張影像
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    # ??
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    # 使用框架
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    # 使用數據集
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    # 使用架構權重
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    # 運用cpu線程數
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    # img導入訓練模型尺寸
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    # 模型訓練權重輸出％步伐
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between saving model weights")
    # epoch % by to compute map
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()
    print(opt)

    # Log to save training process to tensorboard
    logger = Logger("logs")
    # Use the CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    # checkpoints 模型訓練權重輸出資料夾
    os.makedirs("checkpoints/tiny", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    # 記錄訓練影像路徑txt
    train_path = data_config["train"]
    # 記錄驗證影像路徑txt
    valid_path = data_config["valid"]
    # 記錄標籤名稱
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    # weights_init_normal come from utils
    model.apply(weights_init_normal)

    # If specified we start from checkpoint權重讀取選擇
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    # 導入訓練影像路徑 ; 
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    """
    加載數據集 ; 
    每個batch加載幾個圖像 ; 
    每個epoch會重新打亂數據 ; 
    用幾個子線程加載數據 ;  
    如果数据集大小不能被batch size整除，则设置为True后可删除最后一个不完整的batch。如果设为False并且数据集的大小不能被batch size整除，则最后一个batch将更小
    """
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        drop_last=False
    )

    # 訓練權重的優化器
    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    # 開始訓練模型
    for epoch in range(opt.epochs):
        # 調整模行為訓練模式
        model.train()
        start_time = time.time()
        #images size -> torch.Size([batch, 3, 512, 512])
        #target size -> torch.Size([batch, 6])
        #假設有1000筆訓練資料->10 batch->
        #len(dataloader) 迭代100次
        #每個epoch -> batch_i ->0~100
        #batches_done -> 計算batch總次數
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i
            
            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/tiny/yolov3_ckpt_%d.pth" % epoch)
