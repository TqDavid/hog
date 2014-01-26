import cv2, os


class LoadDataSet:

	def __init__(self, dbFolder):
		self.trainFolder = os.path.join(dbFolder, 'Train')
		self.testFolder = os.path.join(dbFolder, 'Test')

	def loadTrainSet(self):
		trainPos = self.list_pos(self.trainFolder)
		trainNeg = self.list_neg(self.trainFolder)
		return trainPos, trainNeg


	def loadTestSet(self):
		testPos = self.list_pos(self.testFolder)
		testNeg = self.list_neg(self.testFolder)
		return testPos, testNeg


	def list_pos(self, folder):
		retList = list()
		pos_folder = os.path.join(folder, 'pos')
		desc_folder = os.path.join(folder, 'annotations')

		for imName in os.listdir(pos_folder):
			image = os.path.join(pos_folder, imName)

			txtName = os.path.join(desc_folder, imName[0:-4] + ".txt")
			retList.append( (image, txtName))

		return retList


	def list_neg(self, folder):
		retList = list()
		neg_folder = os.path.join(folder, 'neg')

		for imName in os.listdir(neg_folder):
			image = os.path.join(neg_folder, imName)
			retList.append(image)

		return retList
