import cv2

from Load import LoadDataSet
from Cut import CutPicture
from Extract import HOG, Extract

import numpy as np


db = LoadDataSet("../../../INRIAPerson")
cut = CutPicture(64.0, 128.0, 3)
hog = HOG(9, (6,6), (3,3))
ex = Extract(hog, cut)

'''
print 'Generiranje znacajki...'
pos, neg = db.loadTrainSet()
X, y = ex.getSamples(pos, neg)

X = np.array(X).astype('float32')
y = np.array(y).astype('float32')

print 'Ucenje...'
model = cv2.SVM() 
params = dict(kernel_type=cv2.SVM_LINEAR, svm_type=cv2.SVM_C_SVC) 
model.train_auto(X, y, None, None, params, 3) #kfold=3 (default: 10)

model.save('../Model/model.svm')
'''
print 'Testiranje...'
model = cv2.SVM()
model.load('../Model/model.svm')
pos = db.loadTestSet()
X = ex.getTestSamples(pos)
X = np.array(X).astype('float32')


true = 0
false = 0

for sample in X:
	if int(model.predict(sample)) == 1:
		true += 1
	else:
		false += 1

print 'N: ', true + false
print 'True: ', true, true / float(true + false)
print 'False: ', false
	
'''
ret = model.predict_all( X )
N = ret.size
true = np.nonzero( ret )[0].size

print 'N: ', ret.size
print 'True: ', true, true / float( N )
print 'False: ', N - true
'''
print 'Gotovo...'
