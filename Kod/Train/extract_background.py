import cv2
import cv2.cv as cv
import numpy as np
import sys


def key_with_max_value(d):
    v=list(d.values())
    k=list(d.keys())
    return k[v.index(max(v))]


if len(sys.argv) != 3:
	print "parametri: video file_za_pozadinu"
	sys.exit()

cap = cv2.VideoCapture(sys.argv[0])
frames = []

while True:
	ret,im = cap.read()
	if im == None:
		break
	frames.append(im)


height, width, depth = frames[0].shape

# ovo je samo inicijalizacija varijable background,
# sa zeros ne radi dobro
background = frames[0]

for i in range(height-1):
	for j in range(width-1):
		avg = {}
		for k in range(len(frames)-1):
			if not frames[k][i][j][0] in avg.keys():
				avg[frames[k][i][j][0]] = 1
			else:
				avg[frames[k][i][j][0]] += 1

		background[i][j][:] = key_with_max_value(avg)

cv2.imwrite(sys.argv[1], background)