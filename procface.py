import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.1289,
	help="minimum probability to filter weak detections")
	#default=0.1289
args = vars(ap.parse_args())
count=0
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()
for i in range(0, detections.shape[2]):
	confidence = detections[0, 0, i, 2]
	if confidence > args["confidence"]:		
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int") 
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		count=count+1
if (count-1)>0:
	print "%s FACES"%(count-1)
else:
	print "NO FACE"

