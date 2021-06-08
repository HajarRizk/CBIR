import cv2
import imutils
from matplotlib import pyplot as plt

class ColorDescriptor:
    def __init__(self, bins):
        #Storing number of bins for histogram
        self.bins = bins
        
    def describe(self, image):
        #Convert the image into hsv and initialize the features to quantify the image
        image = image.astype('uint8')
        features = []


        hist = self.histogram(image)
        features.extend(hist)
        
        #Return the feature vector
        return features
    
    def histogram(self,image):
        
        #Extract a 3-D color histogram using the number of bins supplied
        hist = cv2.calcHist([image], [0, 1, 2], None, self.bins, [0, 256, 0, 256, 0, 256])
        
        #Normalize the histogram
        if imutils.is_cv2():
            #For openCV version 2.4
            hist = cv2.normalize(hist).flatten()
        else:
            #For openCV version 3+
            hist = cv2.normalize(hist, hist).flatten()
            
        #Returning histogram
        return hist


#image = cv2.imread("queries/a140032.jpg")
#test = ColorDescriptor((8,12,10))
#plt.imshow(imutils.opencv2matplotlib(image))
#plt.show()
#value = test.describe(image)
#print(len(value))