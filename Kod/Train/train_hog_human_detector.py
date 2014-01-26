import cv2

from Load import LoadDataSet
from Cut import CutPicture
from Extract import HOG, Extract

import numpy as np
import sys, os

if len(sys.argv) != 3:
	print 'Usage: python train_hog_human_detectory.py [lokacija baze] [mode]'
	print 'mode: 0 - train, 1 - test'
	sys.exit(0)

if os.path.isdir(sys.argv[1]) == False:
	print 'Ne postoji direktorij', sys.argv[1] 
	sys.exit(0)

db = LoadDataSet(sys.argv[1])
cut = CutPicture(64.0, 128.0, 3)
hog = HOG(9, (6,6), (3,3))
ex = Extract(hog, cut)


if int(sys.argv[ 2 ]) == 0:
	print 'Generiranje znacajki...'
	pos, neg = db.loadTrainSet()
	
	lPos = 1178
	lNeg = 1359

	#X, y = ex.getSamples(pos, neg, lPos, lNeg)
	
	X, y = ex.getSamples(pos, neg)
	
	X = np.array(X).astype('float32')
	y = np.array(y).astype('float32')
	
	noPos = np.sum(y == 1.0)
	noNeg = np.sum(y == 0.0)
	print "Train pos: " + str(noPos)
	print "Train neg: " + str(noNeg)

	print 'Ucenje...'
	model = cv2.SVM() 
	params = dict(kernel_type=cv2.SVM_LINEAR, svm_type=cv2.SVM_C_SVC) 
	model.train_auto(X, y, None, None, params, 3) #kfold=3 (default: 10)
	model.save('../Model/model.svm')

elif int(sys.argv[ 2 ]) == 1:
	print 'Testiranje...'
	model = cv2.SVM()
	model.load('../Model/model.svm')

	print "Model je ucitan..."

	pos, neg = db.loadTestSet()
	X, y = ex.getSamples(pos, neg)

	X = np.array(X).astype('float32')
	y = np.array(y).astype('float32')

	noPos = np.sum(y == 1.0)
	noNeg = np.sum(y == 0.0)
	print "Test pos: " + str(noPos)
	print "Test neg: " + str(noNeg)

	TP = 0
	TF = 0
	FP = 0
	FN = 0
	for i in range(len(X)):
		ret = int(model.predict(X[i].astype('float32')))
		if ret == 1 and y[i] == 1:
			TP += 1
		elif ret == 0 and y[i] == 0:
			TF += 1
		elif ret == 1 and y[i] == 0:
			FP += 1
		elif ret == 0 and y[i] == 1:
			FN += 1

	N = len(X)
	print 'N: ', N
	print 'Accuracy: ', (TP+TF) / float( N )
	print 'True: ', (TP+TF)
	print 'False: ', (N-TP-TF)
	print "TP: ", TP
	print "TF: ", TF
	print "FP: ", FP
	print "FN: ", FN
	print 'END...'

else:
	print 'Mode', sys.argv[ 2 ], 'ne postoji'