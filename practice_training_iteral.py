from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
import argparse


parser = argparse.ArgumentParser()
# 訓練步伐
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
# 一次導入幾張影像
parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
# ??
parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
# 使用數據集
parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
# 運用cpu線程數
parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
# img導入訓練模型尺寸
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
opt = parser.parse_args()
print(opt)


# 記錄訓練影像路徑txt
data_config = parse_data_config(opt.data_config)
# 記錄訓練影像路徑txt
train_path = data_config["train"]
# 記錄驗證影像路徑txt
valid_path = data_config["valid"]
# 記錄標籤名稱
class_names = load_classes(data_config["names"])
    
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
# 開始訓練模型
epoch = 0
while epoch < 1:
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        print("\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader)))
        print("batch_i : " , batch_i)
        print("imgs : " , imgs.size())
        print("targets : " , targets.size())
        print("len(dataloader) : ",len(dataloader))
        batches_done = len(dataloader) * epoch + batch_i
        print("batches_done : ",batches_done)
        break
    epoch+=1
        