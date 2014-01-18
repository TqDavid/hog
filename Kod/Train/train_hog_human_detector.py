import cv2

from Load import LoadDataSet
from Cut import CutPicture
from Extract import HOG, Extract

import numpy as np


db = LoadDataSet("../../..//INRIAPerson")
cut = CutPicture(64.0, 128.0, 3)
hog = HOG(9, (6,6), (3,3))
ex = Extract(hog, cut)


# Ovaj dio radi ucenje
pos, neg = db.loadTrainSet()
samples, classes = ex.getSamples(pos, neg)

samples = np.array(samples).astype('float32')
classes = np.array(classes).astype('float32')

print 'Ucenje'
model = cv2.SVM()
model.train(samples, classes)
#model.save('Model/model1.svm')
"""
 self.est = cv2.SVM()
        params = dict(kernel_type=cv2.SVM_LINEAR, svm_type=cv2.SVM_C_SVC)
        self.est.train_auto(X, y, None, None, params, 3) #kfold=3 (default: 10)
"""
# ovaj dio radi testiranje 

#model = cv2.SVM()
#model.load('Model/model.svm')

print 'Testiranje'
pos = db.loadTestSet()
samples = ex.getTestSamples(pos)

true = 0
false = 0
N = len(samples)

for s in samples:
	if int(model.predict(np.array(s).astype('float32'))) == 1:
		true = true + 1
	else:
		false = false + 1

print 'N: ', N
print 'True: ', true
print 'False ', false
