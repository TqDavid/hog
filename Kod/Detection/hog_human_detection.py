from Load import LoadVideo
from Preprocess import Preprocess
import cv2

video = LoadVideo("/home/dino/Desktop/test_hog/video.avi")
preprocess = Preprocess()

for im in video.get_frames():
	k = preprocess.getRectObjectFromImage(im)
	for i in k:
		cv2.imshow("test_obj", i.image)	
	cv2.waitKey(1000)