from Common import SubImage
import cv2

class Preprocess:

	#Kreiranje objekta (automat stanja) koji mice povrsinu
	def __init__(self, size):
		self.background = cv2.BackgroundSubtractorMOG()
		#self.background = None#cv2.imread("test.png")
		#self.background = cv2.cvtColor(self.background, cv2.COLOR_BGR2GRAY)
		
		self.height = size[1]
		self.width = size[0]

	#Funkcija za rezanje pravokutnika iz slike
	def cutImage(self, image, rectangle):
		ymin = rectangle[1]
		ymax = ymin + rectangle[3]
		xmin = rectangle[0]
		xmax = xmin + rectangle[2]
		
		width = rectangle[2]
		xcenter = round((xmax + xmin) / 2)
		height = rectangle[3]
		ycenter = round((ymax + ymin) / 2)
		
		if height / width < self.height / self.width:
				height = self.height / self.width * width
				ymax = min(image.shape[0] - 1, ycenter + round(height / 2))
				ymin = max(0, ycenter - round(height / 2))
		else:
				width = self.width / self.height * height
				xmax = min(image.shape[1] - 1, xcenter + round(width / 2))
				xmin = max(0, xcenter - round(width / 2))
		
		if ymin == ymax or xmin == xmax:
			return None

		tmp = image[ymin:ymax, xmin:xmax]
		return tmp


	#Vraca listu objekata SubImage (pravokutnike u kojima su objeki)
	def getRectObjectFromImage(self, frame):
		
		returnList = list()
		
		#Pretvaranje u sivu i ucenje povrsine
		#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#frame = cv2.equalizeHist(frame) 
		"""
		if self.background is None:
			self.background = frame

		thresh_d = 0
		thresh_u = 0
		
		if len(frame.shape) == 3:
			frame_b = frame[:,:,0]
			frame_g = frame[:,:,1]
			frame_r = frame[:,:,2]
		
			background_b = self.background[:,:,0]
			background_g = self.background[:,:,1]
			background_r = self.background[:,:,2]

			diff_b = cv2.absdiff(background_b, frame_b)
			diff_g = cv2.absdiff(background_g, frame_g)
			diff_r = cv2.absdiff(background_r, frame_r)

			diff = diff_r + diff_g + diff_b
			thresh_d = 100
			thresh_u = 765
		else:
			diff = cv2.absdiff(self.backgorund, frame)
			thresh_d = 80
			thresh_u = 255

		tmp, processImage = cv2.threshold(diff, 100, 765, cv2.THRESH_BINARY) 

		"""
		
		processImage = self.background.apply(frame)
		
		#processImage = cv2.GaussianBlur(processImage, (11,11), 0.5)
		
		
		#Trazenje kontura i detekcija objekata
		contours, hierarchy = cv2.findContours(processImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		cv2.drawContours(processImage, contours, -1, (255), thickness=10)
		cv2.imshow("a", processImage)
		
		contours, hierarchy = cv2.findContours(processImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		#Izdvajanje objekata
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		if len(contours) > 0:
			for i in contours:
				x, y, w, h = cv2.boundingRect(i)
				if w*h < 300:
					continue
				crop = frame[y:y+h,x:x+w]
				crop = cv2.resize(crop, (self.width, self.height))
				returnList.append(SubImage(x, y, w, h, crop))

		return returnList