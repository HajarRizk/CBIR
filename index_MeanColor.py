
import MeanColor as MC

# Used for parsing command line arguments
import argparse
# Used for getting paths of our images
import glob
import cv2

# Initializing our color descriptor
cd = MC.MeanColor()

# open the output index file for writing
output = open("index_MeanColor.csv", "w")

# Using glob to get path of images and go through all of them
for imagePath in glob.glob("dataset" + "/*.jpg"):
    # Get the UID of the image path and load the image
    imageUID = imagePath[imagePath.rfind("/") + 1:]
    image = cv2.imread(imagePath)
    # Using the describe function
    features = cd.describe(image)

    # write the features to a csv file
    features = [str(f) for f in features]
    output.write("%s,%s\n" % (imageUID, ",".join(features)))

# closing the index file
output.close()
