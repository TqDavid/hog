import cv2

from Load import LoadDataSet
from Cut import CutPicture
from Extract import HOG, Extract

import numpy as np
import sys

db = LoadDataSet(sys.argv[1])
cut = CutPicture(64.0, 128.0, 3)
hog = HOG(9, (6,6), (3,3))
ex = Extract(hog, cut)


print 'Generiranje znacajki...'
pos, neg = db.loadTrainSet()
X, y = ex.getSamples(pos, neg)

X = np.array(X).astype('float32')
y = np.array(y).astype('float32')

print "Train pos: " + str(len(pos))
print "Train neg: " + str(len(neg))

print 'Ucenje...'
model = cv2.SVM() 
params = dict(kernel_type=cv2.SVM_LINEAR, svm_type=cv2.SVM_C_SVC) 
#model.train_auto(X, y, None, None, params, 3) #kfold=3 (default: 10)

#model.save('../Model/model.svm')

print 'Testiranje...'
model = cv2.SVM()
model.load('../Model/model.svm')

print "Model je ucitan..."

pos, neg = db.loadTestSet()
X, y = ex.getSamples(pos, neg)
X = np.array(X).astype('float32')
y = np.array(y).astype('float32')

print "Test pos: " + str(len(pos))
print "Test neg: " + str(len(neg))

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