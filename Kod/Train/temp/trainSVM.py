#!/usr/bin/env python
 # -*- coding: utf-8 -*-

import sys
import cv2
import re
import numpy as np
import os, sys
from skimage.feature import hog
import csv, random

from numpy import array

def readCSV(inFile):
	readerObj = csv.reader(open(inFile, 'rb'))
	first = True
	header = readerObj.next()
	first = readerObj.next()

	features = array([first[0:-1]]).astype('float32')
	classes = array([first[-1]]).astype('float32')

	for row in readerObj:
		classes = np.vstack((classes, array([row[-1]]).astype('float32')))
		features = np.vstack((features, array([row[0:-1]]).astype('float32')))
	
	return features, classes

def trainSVM(samples, classes):
	model = cv2.SVM()
	model.train(samples, classes)
	return model

def main():
	#predvideni parametri
	#1: putanja do csv datoteke
	#2: datoteka u koju spremamo model
	try:
		samples, classes = readCSV(sys.argv[ 1 ])
		model = trainSVM(samples, classes)
		model.save(sys.argv[ 2 ])

	except:
		print 'Doslo je do greske! :)'

if __name__ == "__main__":
	main()
