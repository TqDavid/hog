from Load import LoadVideo
from Preprocess import Preprocess
from Detection import Detection
import cv2

video = LoadVideo("../../video.avi")
#video = LoadVideo(0) #za web kameru ili ugradjenu na racunalu

detection = Detection("model.svm")

preprocess = Preprocess((64, 128))

for im in video.get_frames():
	k = preprocess.getRectObjectFromImage(im)
	for i in k:
		if ( detection.get_class(i.image) == 1):
			cv2.rectangle(im, (i.x, i.y), (i.x+i.w, i.y+i.h), (255,0,0), thickness=2)
		
	im = cv2.resize(im, (640,480))
	cv2.imshow("test_obj", im)
	cv2.waitKey(1)