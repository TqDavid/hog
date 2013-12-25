import cv2

class LoadVideo:

	def __init__(self, filename):
		self.videoObj = self.load_video(filename)
		
	#Generiranje slika iz videa
	def get_frames(self):
		ret,im = self.videoObj.read()
		while im is not None:
			yield im
			ret,im = self.videoObj.read()

	#Ucitavanje videa
	def load_video(self, filename):
		return cv2.VideoCapture(filename)


