from Common import SubImage
import cv2
import numpy as np

class Preprocess:

	#Kreiranje objekta (automat stanja) koji mice povrsinu
	def __init__(self, size):
		self.background = cv2.BackgroundSubtractorMOG()
		#self.background = None#cv2.imread("test.png")
		#self.background = cv2.cvtColor(self.background, cv2.COLOR_BGR2GRAY)
		
		self.height = size[1]
		self.width = size[0]
		self.test = None

	#Vraca listu objekata SubImage (pravokutnike u kojima su objeki)
	def getRectObjectFromImage(self, frame):
		
		returnList = list()
		
		diff = None
		if self.test is None:
			self.test = np.copy(frame)
		else:
			diff = cv2.absdiff(self.test, frame)
			self.test = np.copy(frame)

		if diff is not None:
			diff =  cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
			tmp, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY) 
			

		processImage = self.background.apply(frame)
		
		if diff is not None:
			processImage = processImage + diff

		kernel = np.ones((7,3),np.uint8)
		processImage = cv2.dilate(processImage, kernel, iterations = 5)

		cv2.imshow("Foreground", processImage)

		#Trazenje kontura i detekcija objekata
		contours, hierarchy = cv2.findContours(processImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		
		cv2.drawContours(processImage, contours, -1, (255), thickness=-1)
		
		
		contours, hierarchy = cv2.findContours(processImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		#Izdvajanje objekata
		
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		windows = []
		
		if len(contours) > 0:
			for i in contours:
				x, y, w, h = cv2.boundingRect(i)

				if w > h:
					continue

				if w*h < 300:
					continue

				if w*h > 128*64*2:
					continue

				windows.append((x,y,w,h))
				
		for i in windows:
			x, y, w, h = i
			crop = frame[y:y+h,x:x+w]
			crop = cv2.resize(crop, (self.width, self.height))
			returnList.append(SubImage(x, y, w, h, crop))

		return returnList

	#Provjeravanje da li jedan od kuteva pravokutnika nalazi u im
	def rectIn(self, x, y, w, h, im):

		if ( x >= im.x and x <= im.x+im.w and y >= im.y and y <= im.y+h ):
			return True
		
		if ( x+w >= im.x and x+w <= im.x+im.w and y >= im.y and y <= im.y+h ):
			return True
		
		if ( x >= im.x and x <= im.x+im.w and y+h >= im.y and y+h <= im.y+h ):
			return True

		if ( x+w >= im.x and x+w <= im.x+im.w and y+h >= im.y and y+h <= im.y+h ):
			return True

		return False

	#Ispitivanje da li (x,y,w,h) pravokutnik nalazi u predhodnima
	def notContain(self, x, y, w, h, humanImages):
		for im in humanImages:
			wIm = im.x + im.w / 2
			hIm = im.y + im.h / 2
			wMe = x + w / 2;
			hMe = y + h / 2;

			distance = ((wIm-wMe)**2 + (hIm - hMe)**2)

			if  distance < 100:
				return False
			
			if self.rectIn(x,y,w,h,im):
				return False
		
		return True