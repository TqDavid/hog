#!/usr/bin/env python
 # -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import os, sys

import getFeatures
from numpy import array

#metoda koja izreze ljude iz pozitiva 
#vrati vise slika

def readPicPos(inPic, inDesc):
	image = cv2.imread(inPic)
	if len(image.shape) == 3:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	txtFile = open(inDesc).read()

	return getFeatures.getTrainingExample(image, txtFile)

#dobijemo neku sliku, ucitamo, skaliramo i RGB2GRAY
def readPic(inPic):
	image = cv2.imread(inPic)
	if len(image.shape) == 3:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	return cv2.resize(image, (64, 128))	


# metoda koja ucita model SVM-a
# za 9, (6 6), (3 3) je datoteka velika 400 MB
def readModel(inModelTxt):
	model = cv2.SVM()
	model.load(inModelTxt)

	return model

def main():
	#predvideni parametri
	#1: putanja do opisa modela
	#2: putanja do slike

	try:
		model = readModel(sys.argv[ 1 ])
		ORN = 9 ; PIX = (6,6); BLOCK = (3,3) # parametri za HOG
		
		'''
		pics = readPicPos(sys.argv[ 2 ], sys.argv[ 3 ]) 
		#idemo po slikama i ispisujemo izlaz nakon primjene modela 
		for p in pics:
			descriptor = getFeatures.getHogDescriptor(p, ORN, PIX, BLOCK)
			f = array(descriptor).astype('float32')
			print model.predict(f)
		'''
		
		img = readPic(sys.argv[ 2 ])
		descriptor = getFeatures.getHogDescriptor(img, ORN, PIX, BLOCK)
		f = array(descriptor).astype('float32')
		print model.predict(f)

	except:
		print 'Doslo je do greske! :)'

if __name__ == "__main__":
	main()
