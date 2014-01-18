import cv2, os, re, random

class CutPicture:

	def __init__(self, width, height, numNegCuts):
		self.WIDTH = width
		self.HEIGHT = height
		self.negCuts = numNegCuts


	def getPeople(self, img, desc):
		lines = open(desc).read().split('\n')
		image = cv2.imread(img)
		if len(image.shape) == 3:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		people = []
		expr = re.compile("\((\d+), (\d+)\) - \((\d+), (\d+)\)")
		for line in lines:
			match = expr.search(line)
			
			if match:
				xmin = float(match.group(1))
				ymin = float(match.group(2))
				xmax = float(match.group(3))
				ymax = float(match.group(4))

				width = xmax - xmin
				xcenter = round((xmax + xmin) / 2)
				height = ymax - ymin
				ycenter = round((ymax + ymin) / 2)

				if height / width < self.HEIGHT / self.WIDTH:
					height = self.HEIGHT / self.WIDTH * width
					ymax = min(image.shape[0] - 1, ycenter + round(height / 2))
					ymin = max(0, ycenter - round(height / 2))
				else:
					width = self.WIDTH / self.HEIGHT * height
					xmax = min(image.shape[1] - 1, xcenter + round(width / 2))
					xmin = max(0, xcenter - round(width / 2))

				tmp = cv2.resize (image[ymin:ymax, xmin:xmax], (int(self.WIDTH), int(self.HEIGHT)))
				people.append(tmp)
				people.append(cv2.flip(tmp, 1)) # dodatno jos i zrcaljena slika po y osi
		return people



	def getPatches(self, img):
		patches = list()
		image = cv2.imread(img)
		if len(image.shape) == 3:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		for i in range(0, self.negCuts):
			y = random.randint(0, image.shape[ 0 ] - self.HEIGHT)
			x = random.randint(0, image.shape[ 1 ] - self.WIDTH)
						
			patches.append(cv2.resize(image[y:y+self.HEIGHT, x:x+self.WIDTH], (int(self.WIDTH), int(self.HEIGHT))))
		return patches