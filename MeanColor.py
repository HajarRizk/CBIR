import numpy as np
import cv2



class MeanColor:


    def describe(self, image):
        # Convert the image into hsv and initialize the features to quantify the image
        image = image.astype('uint8')
        RGBmean = []



        # Obtaining The mean of each channel
        (h, w) = image.shape[:2]
        number_of_pixels = h*w
        red = image[:,:,0]
        blue= image[:,:,1]
        green = image[:,:,2]
        RGBmean.append(np.sum(red)/number_of_pixels)
        RGBmean.append(np.sum(blue)/number_of_pixels)
        RGBmean.append(np.sum(green)/number_of_pixels)

        # Return the feature vector
        return RGBmean



#image = cv2.imread("queries/art294.jpg")
#test = MeanColor()
#value = test.describe(image)
#print(value)