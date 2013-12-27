import cv2
from skimage.feature import hog
import numpy as np

class Detection:

	def __init__(self, filename):

		self.model = cv2.SVM()
		self.model.load(filename)

	def get_class(self, image):
		features = hog(image, orientations=9, pixels_per_cell=(6,6), cells_per_block=(3,3), visualise=False)
		ret = self.model.predict(features.astype('float32'))
		return int(ret)