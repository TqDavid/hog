#!/usr/bin/env python
 # -*- coding: utf-8 -*-

import sys
import cv2
import re
import numpy as np
import os, sys
from skimage.feature import hog
import csv, random

HEIGHT = 128.0
WIDTH = 64.0

DEFAULT_POS = 1 
DEFAULT_NEG = 0
DEFAULT_CUTS = 3


def getTrainingExample(image, description):
	lines = description.split('\n')
	people = []
	expr = re.compile("\((\d+), (\d+)\) - \((\d+), (\d+)\)")
	for line in lines:
		match = expr.search(line)
		##print line
		if match:
			xmin = float(match.group(1))
			ymin = float(match.group(2))
			xmax = float(match.group(3))
			ymax = float(match.group(4))

			width = xmax - xmin
			xcenter = round((xmax + xmin) / 2)
			height = ymax - ymin
			ycenter = round((ymax + ymin) / 2)

			if height / width < HEIGHT / WIDTH:
				height = HEIGHT / WIDTH * width
				ymax = min(image.shape[0] - 1, ycenter + round(height / 2))
				ymin = max(0, ycenter - round(height / 2))
			else:
				width = WIDTH / HEIGHT * height
				xmax = min(image.shape[1] - 1, xcenter + round(width / 2))
				xmin = max(0, xcenter - round(width / 2))

			tmp = cv2.resize (image[ymin:ymax, xmin:xmax], (int(WIDTH), int(HEIGHT)))
			people.append(tmp)
	return people


def getTrainingSet(picFolder, descFolder):
	imageSet = []
	for imName in os.listdir(picFolder):
		image = cv2.imread(os.path.join(picFolder, imName))
		if len(image.shape) == 3:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		txtName = os.path.join(descFolder, imName[0:-4] + ".txt")
		txtFile = open(txtName)
		text = txtFile.read()
		images = getTrainingExample(image, text)
		imageSet = imageSet + images
	return imageSet


# REZANJE NEGATIVA

#metoda koja prima putanju do foldera negativa
#poziva metodu za rezanje podslika
#vrati skup slika iz kojih se uzimaju znacajke
def getNegImageSet(picFolder):
	imageSet = list()
	for imName in os.listdir(picFolder):
		image = cv2.imread(os.path.join(picFolder, imName))
		if len(image.shape) == 3:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		
		imgCuts = cutRandomSubImage(image)
		imageSet = imageSet + imgCuts
	return imageSet


#metoda koja slucajno reže podslike 
#vraca skup podslika 
def cutRandomSubImage(image):
	ret = list()
	#tu sada random generiramo jedan broj tako da nam je u dobrom području i onda izrežemo 
	# taj dio slike 
	for i in range(0, DEFAULT_CUTS):
		y = random.randint(0, image.shape[ 0 ] - HEIGHT)
		x = random.randint(0, image.shape[ 1 ] - WIDTH)
		ret.append(cv2.resize(image[y:y+HEIGHT, x:x+WIDTH], (int(WIDTH), int(HEIGHT))))

	return ret

#

# RAD s csv datotekama

#zapisuje header csv datoteke
#params: csv writer Obj, broj elementa u opisniku
def writeHeader(writerObj, N):
	header = ['f' + str(x) for x in range(0, N)]; header.append("class")
	writerObj.writerow(header)

#zapisuje redak csv datoteke
#params: csv writerObj, opisnik, klasa 
def writeHogDescriptor(writerObj, fd, c):
	row = fd.tolist(); row.append(c)
	writerObj.writerow(row)

#zapisuje csv file 
#prima lokaciju izlaznog fajla, opisnike pozitiva i negativa
def writeCSVFile(outFile, pos, neg):
	writerObj = csv.writer(open(outFile, 'wb'))

	for i in range(0, len(pos)):
		if i == 0: writeHeader(writerObj, len(pos[ 0 ]))
		writeHogDescriptor(writerObj, pos[ i ], DEFAULT_POS)

	for j in range(0, len(neg)):
		writeHogDescriptor(writerObj, neg[ j ], DEFAULT_NEG)

#

#params: ulazna slika, (int)broj_binova, (tupple)cells_per_block, (tupple)pixels_per_cell
#primjer getHogDesctipro(image, 8, (16,16), (1,1))
def getHogDescriptor(inImage, ORN, PIX, BLOCK):
	#visualse = False, znaci da nam treba samo flattened array s deskriptorom
	fd = hog(inImage, orientations = ORN, pixels_per_cell= PIX, 
		cells_per_block= BLOCK, visualise=False)
	return fd


#params: skup slika,  i parametri za HOG: bins, pixels_per_cell i cells_per_block 
#vraca listu opisnika
def getFeatures(images, ORN, PIX, BLOCK):
	ret = list()
	for i in range(0, len(images)):
		fd = getHogDescriptor(images[ i ], ORN, PIX, BLOCK)
		ret.append(fd)
	return ret

#glavna funkcija
def getAllFeatures(picFolder, descFolder, csvFile, negPicFolder):
	ORN = 8 ; PIX = (16,16); BLOCK = (1,1) # parametri za HOG
	
	#ovo bude izrezalo pozitive i uzelo znacajke
	pos_images = getTrainingSet(picFolder, descFolder)
	pos_features = getFeatures(pos_images, ORN, PIX, BLOCK)

	#ovo bude izrezalo negative i uzelo znacajke
	neg_images = getNegImageSet(negPicFolder)
	neg_features = getFeatures(neg_images, ORN, PIX, BLOCK)

	writeCSVFile(csvFile, pos_features, neg_features)
	

def main():
	#predvideni parametri
	#1: pozitivi_slike folder	2: opisnici pozitiva folder
	#3: negativi_slike 			4: izlazna csv datoteka 		

	try:
		getAllFeatures(sys.argv[ 1 ], sys.argv[ 2 ], sys.argv[ 4 ], sys.argv[ 3 ])
	except:
		print 'Doslo je do greske! :)'

if __name__ == "__main__":
	main()
