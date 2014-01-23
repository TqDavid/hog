from Load import LoadVideo
from Preprocess import Preprocess
from Detection import Detection
from Tracking import Tracking
from Common import SubImage
import cv2
import numpy as np

def rectIn(x, y, w, h, im):

	if ( x >= im.x and x <= im.x+im.w and y >= im.y and y <= im.y+h ):
		return True
	
	if ( x+w >= im.x and x+w <= im.x+im.w and y >= im.y and y <= im.y+h ):
		return True
	
	if ( x >= im.x and x <= im.x+im.w and y+h >= im.y and y+h <= im.y+h ):
		return True

	if ( x+w >= im.x and x+w <= im.x+im.w and y+h >= im.y and y+h <= im.y+h ):
		return True

	return False


def notContain(x, y, w, h, humanImages):
	for im in humanImages:
		wIm = im.x + im.w / 2
		hIm = im.y + im.h / 2
		wMe = x + w / 2;
		hMe = y + h / 2;

		distance = ((wIm-wMe)**2 + (hIm - hMe)**2)
		
		if  distance < 100:
			return False
		
		if rectIn(x,y,w,h,im):
			return False
	
	return True

video = LoadVideo("../../video.avi")
#video = LoadVideo(0) #za web kameru ili ugradjenu na racunalu
#video = LoadVideo("/home/dino/Desktop/test_hog/test.mpg")
#video = LoadVideo("/home/dino/Desktop/test1.avi")

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
		#cv2.rectangle(im, (i.x, i.y), (i.x+i.w, i.y+i.h), (0,255,0), thickness=1)
	humanImages = newHumanImages

	if cnt % 2 == 0:
		newHumanImages = []
		rectangles = preprocess.getRectObjectFromImage(im)
		for i in rectangles:
			if ( detection.get_class(i.image) == 1 ):
				#if notContain(i.x, i.y, i.w, i.h, humanImages):
				humanImg = np.copy(im[i.y:i.y+i.h,i.x:i.x+i.w])
				newHumanImages.append(SubImage(i.x,i.y,i.w,i.h, humanImg))
			#cv2.rectangle(im, (i.x, i.y), (i.x+i.w, i.y+i.h), (255,0,0), thickness=1)
				#cv2.rectangle(im, (i.x, i.y), (i.x+i.w, i.y+i.h), (0,0,255), thickness=1)

		humanImages = humanImages + newHumanImages

	newHumanImages = []
	for i in humanImages:
		if notContain(i.x, i.y, i.w, i.h, newHumanImages):
			humanImg = np.copy(im[i.y:i.y+i.h,i.x:i.x+i.w])
			newHumanImages.append(SubImage(i.x,i.y,i.w,i.h, humanImg))
			cv2.rectangle(im, (i.x, i.y), (i.x+i.w, i.y+i.h), (0,255,0), thickness=1)
	
	humanImages = newHumanImages

	"""
	if cnt % 2 != 0:
		newHumanImages = []
		rectangles = Tracking.findSubImages(im, humanImages)
		for i in rectangles:
			humanImg = np.copy(im[i.y:i.y+i.h,i.x:i.x+i.w])
			newHumanImages.append(SubImage(i.x,i.y,i.w,i.h, humanImg))
			cv2.rectangle(im, (i.x, i.y), (i.x+i.w, i.y+i.h), (0,255,0), thickness=1)
		humanImages = newHumanImages
	else:
		newHumanImages = []
		rectangles = preprocess.getRectObjectFromImage(im)
		for i in rectangles:
			if ( detection.get_class(i.image) == 1 ):
				if notContain(i.x, i.y, i.w, i.h, humanImages):
					humanImg = np.copy(im[i.y:i.y+i.h,i.x:i.x+i.w])
					newHumanImages.append(SubImage(i.x,i.y,i.w,i.h, humanImg))
					cv2.rectangle(im, (i.x, i.y), (i.x+i.w, i.y+i.h), (255,0,0), thickness=1)
				cv2.rectangle(im, (i.x, i.y), (i.x+i.w, i.y+i.h), (0,0,255), thickness=1)

		humanImages += newHumanImages
	"""

	im = cv2.resize(im, (640,480))
	
	cv2.imshow("test_obj", im)
	cv2.waitKey(1)
	cnt += 1