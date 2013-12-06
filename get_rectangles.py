import cv2
import cv2.cv as cv
import numpy as np
import sys

MIN = 5

# vraca relevante okvire za trenutni frame
def getRectangles(frame, background, height, width):
	rectangles = []

	foreground = cv2.absdiff(frame, background)
	tmp, binary = cv2.threshold(foreground, 30, 255, cv2.THRESH_BINARY)

	mask = np.zeros((height+2, width+2), np.uint8)
	image = cv2.medianBlur(binary, 5)

	seed_point = np.argwhere( image == 255 )

	if seed_point.size == 0:
		return rectangles

	retval, rect = cv2.floodFill(image, mask,  (seed_point[0][1], seed_point[0][0]), (0,244,0))
	# cv2.rectangle(image, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (254,0,0))

	if rect[2] > MIN and rect[3] > MIN:
		rectangles.append(rect)
		cv2.rectangle(image, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (254,0,0))

	while np.argwhere( image == 255 ).size > 0:
		seed_point = np.argwhere( image == 255 )
		retval, rect = cv2.floodFill(image, mask,  (seed_point[0][1], seed_point[0][0]), (0,244,0))
		# cv2.rectangle(image, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (254,0,0))

		if rect[2] > MIN and rect[3] > MIN:
			rectangles.append(rect)
			cv2.rectangle(image, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (254,0,0))

	cv2.imshow('flooded', image)
	key = cv2.waitKey()
	return rectangles


if len(sys.argv) != 3:
	print "parametri: video pozadina"
	sys.exit()

cap = cv2.VideoCapture(sys.argv[1])

frames = []
while True:
	ret,im = cap.read()
	if im == None:
		break
	frames.append(im)

height, width, depth = frames[0].shape
background = cv2.imread(sys.argv[2])

rectangles = []

for i in range(len(frames)-1):
	rectangles += getRectangles(frames[i], background, height, width)

# print len(rectangles)