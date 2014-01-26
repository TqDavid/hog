from Load import LoadVideo
from Preprocess import Preprocess
from Detection import Detection
from Tracking import Tracking
from Common import SubImage
import cv2
import numpy as np
import sys

fileName = sys.argv[1]
if len(fileName) == 1:
	fileName = int(fileName)

video = LoadVideo(fileName)

detection = Detection("model.svm")

preprocess = Preprocess((64, 128))

humanImages = []
cnt = 0

for im in video.get_frames():

	newHumanImages = []
	rectangles = Tracking.findSubImages(im, humanImages)
	for i in rectangles:
		humanImg = np.copy(im[i.y:i.y+i.h,i.x:i.x+i.w])
		newHumanImages.append(SubImage(i.x,i.y,i.w,i.h, humanImg))
	humanImages = newHumanImages

	if cnt % 1 == 0:
		newHumanImages = []
		rectangles = preprocess.getRectObjectFromImage(im)
		for i in rectangles:
			
			if ( detection.get_class(i.image) == 1 ):
				humanImg = np.copy(im[i.y:i.y+i.h,i.x:i.x+i.w])
				newHumanImages.append(SubImage(i.x,i.y,i.w,i.h, humanImg))
		humanImages = newHumanImages + humanImages

	newHumanImages = []
	for i in humanImages:
		if preprocess.notContain(i.x, i.y, i.w, i.h, newHumanImages):
			humanImg = np.copy(im[i.y:i.y+i.h,i.x:i.x+i.w])
			newHumanImages.append(SubImage(i.x,i.y,i.w,i.h, humanImg))
			cv2.rectangle(im, (i.x, i.y), (i.x+i.w, i.y+i.h), (0,255,0), thickness=1)

	humanImages = newHumanImages
	
	print "Number of people: ", len(humanImages)

	im = cv2.resize(im, (640,480))
	
	cv2.imshow("Main window", im)
	cv2.waitKey(100)
	cnt += 1