# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
class CentroidTracker:
	
    def __init__(self, maxDisappeared=50,maxDistance=10):
        #編列物件序號
        self.nextObjectID = 0
        #物件存在單
        self.objects = OrderedDict()
        #物件消失單
        self.disappeared = OrderedDict()
        #最大允許消失張數
        self.maxDisappeared = maxDisappeared
        #最大允許差離距離
        self.maxDistance = maxDistance

    def register(self, centroid):
        # 當有一個新物件進入時
        # 會在物件存在單註冊
        self.objects[self.nextObjectID] = centroid
        # 同時也會在物件消失單註冊,初始定義消失張數0張
        self.disappeared[self.nextObjectID] = 0
        #+1序號為下一個須註冊物件做準備
        self.nextObjectID += 1
    
    def deregister(self, objectID):
		# 解除物件存在
        del self.objects[objectID]
        del self.disappeared[objectID]
    
    def update(self, rects):
		
        # 當rects未出現任何物件資訊之情況
        if len(rects) == 0:
            # 代表所有物件都不存在,我們則把物件消失單上的每個物件都記消失一次
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                # 當消失次數大於設定值則將該物件刪除
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            #回傳當前物件追蹤清單
            return self.objects
        
        # 若rects有物件資訊,將物件整理程矩陣紀錄每個的r,c
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # 計算當前每個方框實際的中心位置
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
        # 如果物件存在單上並沒有紀錄任何物件資訊則代表rects內的物件都是新的物件直接進行註冊
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        # 其他的可能則是rects有當下存在物件名單
        # 物件存在單上則有上一輪紀錄的物件資訊
        # 我們則透過歐幾里得距離去判斷新的物件名單與就的物件名單上重複項為何？
        else:
			#可把self.objects看作舊有存在物件之資訊
            #可把inputCentroids看作當前存在物件之資訊
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
			#計算舊有與當前歐基里得距離D
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
			# in order to perform this matching we must (1) find the
			# smallest value in each row and then (2) sort the row
			# indexes based on their minimum values so that the row
			# with the smallest value is at the *front* of the index
			# list
            rows = D.min(axis=1).argsort()
			# next, we perform a similar process on the columns by
			# finding the smallest value in each column and then
			# sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]
            # in order to determine if we need to update, register,
			# or deregister an object we need to keep track of which
			# of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()
			# loop over the combination of the (row, column) index
			# tuples
            for (row, col) in zip(rows, cols):
				# if we have already examined either the row or
				# column value before, ignore it
				# val
                if row in usedRows or col in usedCols:
                    continue
                #當對應物件距離間距過大
                if D[row, col] > self.maxDistance:
                    continue
                # 如果新舊物件判定為同個,則更新既有物件相關資訊
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
				# indicate that we have examined each of the row and
				# column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)
            # compute both the row and column index we have NOT yet
			# examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            # in the event that the number of object centroids is
			# equal or greater than the number of input centroids
			# we need to check and see if some of these objects have
			# potentially disappeared
            if D.shape[0] >= D.shape[1]:
				# loop over the unused row indexes
                for row in unusedRows:
					# grab the object ID for the corresponding row
					# index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
					# check to see if the number of consecutive
					# frames the object has been marked "disappeared"
					# for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            # otherwise, if the number of input centroids is greater
			# than the number of existing object centroids we need to
			# register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])
		# return the set of trackable objects
        return self.objects



class TrackableObject:
	def __init__(self, objectID, centroid):
		# store the object ID, then initialize a list of centroids
		# using the current centroid
		self.objectID = objectID
		self.centroids = [centroid]
		# initialize a boolean used to indicate if the object has
		# already been counted or not
		self.counted = False