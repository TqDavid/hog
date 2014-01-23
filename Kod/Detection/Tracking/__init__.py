from Common import SubImage
import numpy as np
import cv2

class Tracking:

	@staticmethod
	def findSubImages(image, listSubImage):
		
		returnList = []
		
		for img in listSubImage:
			subImage = img.image
			result = cv2.matchTemplate(image, img.image, cv2.TM_CCOEFF_NORMED)
			y, x = np.unravel_index(result.argmax(), result.shape)
			if result.max() < 0.5 or result.max() > 0.99:
				continue
			w, h = img.w, img.h
			returnList.append(SubImage(x, y, w, h, subImage))

		return returnList