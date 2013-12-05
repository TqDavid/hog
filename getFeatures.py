import sys
import cv2
import re
import numpy as np
import os, sys
from skimage.feature import hog
from skimage import data, color, exposure


HEIGHT = 128.0
WIDTH = 64.0


def getTrainingExample(image, description):
        lines = description.split('\n')
        people = []
        expr = re.compile("\((\d+), (\d+)\) - \((\d+), (\d+)\)")
        for line in lines:
                match = expr.search(line)
                ##print line
                if match:
                        xmin = float(match.group(1))
                        ymin = float(match.group(2))
                        xmax = float(match.group(3))
                        ymax = float(match.group(4))

                        width = xmax - xmin
                        xcenter = round((xmax + xmin) / 2)
                        height = ymax - ymin
                        ycenter = round((ymax + ymin) / 2)

                        if height / width < HEIGHT / WIDTH:
                                height = HEIGHT / WIDTH * width
                                ymax = min(image.shape[0] - 1, ycenter + round(height / 2))
                                ymin = max(0, ycenter - round(height / 2))
                        else:
                                width = WIDTH / HEIGHT * height
                                xmax = min(image.shape[1] - 1, xcenter + round(width / 2))
                                xmin = max(0, xcenter - round(width / 2))
                        
                        tmp = cv2.resize (image[ymin:ymax, xmin:xmax], (int(WIDTH), int(HEIGHT)))
                        people.append(tmp)
        return people


def getTrainingSet(picFolder, descFolder):
        imageSet = []
        for imName in os.listdir(picFolder):
                image = cv2.imread(os.path.join(picFolder, imName))
                if len(image.shape) == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                txtName = os.path.join(descFolder, imName[0:-4] + ".txt")
                txtFile = open(txtName)
                text = txtFile.read()
                images = getTrainingExample(image, text)
                imageSet = imageSet + images
        return imageSet
        
def extractFeatures(image):
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualise=True)
        return hog_image

def main():
        images = getTrainingSet(sys.argv[ 1 ], sys.argv[ 2 ])
        for image in images:
                print extractFeatures(image)
                cv2.imshow('proba', image)
                cv2.waitKey()
        

if __name__ == "__main__":
        main()
