import numpy as np
import cv2
import skimage
import os
import matplotlib
import matplotlib.pyplot as plt
from preprocessing import *

dataDir = "./chest-xray-pneumonia/chest_xray/"
trainDir = dataDir + "train/"
testDir = dataDir + "test/"
valDir = dataDir + "val/"

sampleNormalImageName = "NORMAL/IM-0115-0001.jpeg"

sampleImage = cv2.imread(trainDir+sampleNormalImageName)
# print(sampleImage.shape)
# plt.imshow(sampleImage, cmap='gray')
# plt.show()


grayImage = cv2.cvtColor(sampleImage, cv2.COLOR_BGR2GRAY)
plt.subplot(121)
plt.imshow(grayImage, cmap='gray')

# dst = erosion(grayImage, cv2.MORPH_ELLIPSE, 5)
plt.subplot(122)
# dst = dilate(grayImage, 5, cv2.MORPH_ELLIPSE)

# o = openImage(grayImage, 20)

# c = closeImage(grayImage, 20)

# dst = gradientImage(grayImage, 3)

# dst = dilation(grayImage)

dst = equalizeHist(grayImage, globalHist=False)

# dst = adaptiveEq(grayImage, 0.03)

# dst = contrastStretching(grayImage)

# dst = canny(grayImage, 0.4)

# dst = bitplane(grayImage, 0) + bitplane(grayImage, 1) #+ bitplane(grayImage, 2) + bitplane(grayImage, 3) + bitplane(grayImage, 4)

plt.imshow(dst, cmap='gray')
plt.show()