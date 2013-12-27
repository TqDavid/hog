import cv2

class SubImage:

	def __init__(self, x, y, w, h, im):
		self.x = x
		self.y = y
		self.w = w
		self.h = h
		self.image = im

class Preprocess:

	#Kreiranje objekta (automat stanja) koji mice povrsinu
	def __init__(self, size):
		self.background = cv2.BackgroundSubtractorMOG()
		self.height = size[0]
		self.width = size[1]

	#Vraca listu objekata SubImage (pravokutnike u kojima su objeki)
	def getRectObjectFromImage(self, frame):
		
		returnList = list()
		
		#Pretvaranje u sivu i ucenje povrsine
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		processImage = self.background.apply(frame)

		#Trazenje kontura i detekcija objekata
		contours, hierarchy = cv2.findContours(processImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cv2.drawContours(processImage, contours,-1,(255),thickness=5)
		contours, hierarchy = cv2.findContours(processImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		#Izdvajanje objekata
		if len(contours) > 0:
			for i in contours:
				x, y, w, h = cv2.boundingRect(i)
				crop = frame[y:y+h,x:x+w]
				crop = cv2.resize(crop, (self.height, self.width))
				returnList.append(SubImage(x, y, w, h, crop))

		return returnList