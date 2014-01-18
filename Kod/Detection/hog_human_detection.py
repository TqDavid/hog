from Load import LoadVideo
from Preprocess import Preprocess
from Detection import Detection
from Tracking import Tracking
from Common import SubImage
import cv2
import numpy as np

#video = LoadVideo("../../video.avi")
#video = LoadVideo(0) #za web kameru ili ugradjenu na racunalu
video = LoadVideo("/home/dino/Desktop/test.mpg")
#video = LoadVideo("/home/dino/Desktop/test1.avi")

detection = Detection("model.svm")

preprocess = Preprocess((64, 128))

humanImages = []
cnt = 0

for im in video.get_frames():
	
	if cnt % 10 != 0:
		rectangles = Tracking.findSubImages(im, humanImages)
		for i in rectangles:
			cv2.rectangle(im, (i.x, i.y), (i.x+i.w, i.y+i.h), (0,255,0), thickness=2)
	else:
		humanImages = []
		rectangles = preprocess.getRectObjectFromImage(im)
		for i in rectangles:
			if ( detection.get_class(i.image) == 1 ):
				humanImg = np.copy(im[i.y:i.y+i.h,i.x:i.x+i.w])
				humanImages.append(SubImage(i.x,i.y,i.w,i.h, humanImg))
				cv2.rectangle(im, (i.x, i.y), (i.x+i.w, i.y+i.h), (255,0,0), thickness=2)
				
	im = cv2.resize(im, (640,480))

	cv2.imshow("test_obj", im)
	cv2.waitKey(1)
	cnt += 1