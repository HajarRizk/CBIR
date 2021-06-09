
import ColorDescriptor
import MeanColor
import EdgeDescriptor
import Searcher
import cv2





#intializing the color descriptor
cd = ColorDescriptor.ColorDescriptor((16,16,16))
mc = MeanColor.MeanColor()
edge = EdgeDescriptor.Edge()

#loading the query image and describe it
query1 = cv2.imread("queries/9555824387_32b151e9b0_m.jpg") #path of the uploaded picture
query = cv2.resize(query1,(300,300))
cv2.imshow("Query",query)
Method = "ColorDescriptor" #Method taken from user

#performing the search Methods: 1:"MeanColor"->DB:index.csv 2:"ColorDescriptor"->DB:index_MeanColor.csv 3:"Edge"->DB:'index_edge.csv'
if Method=="ColorDescriptor":
  queryFeatures = cd.describe(query)
  s1 = Searcher.Searcher('index.csv')
  results = s1.search(queryFeatures,"ColorDescriptor")
elif Method=="MeanColor":
    queryFeatures = mc.describe(query)
    s1 = Searcher.Searcher('index_MeanColor.csv')
    results = s1.search(queryFeatures,"MeanColor")
elif Method=="Edge":
    queryFeatures = edge.histogram(query)
    s1 = Searcher.Searcher('index_edge.csv')
    results = s1.search(queryFeatures, "Edge")





#displaying the query
cv2.imshow("Query",query)
cv2.waitKey(0)

#loop over the results

for (score, resultID) in results:
    #load the result image and display it
    result1 = cv2.imread(resultID)
    result = cv2.resize(result1,(300,300))
    cv2.imshow("Result",result)
    cv2.waitKey(0)
