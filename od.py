
# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import cvlib as cv
from cvlib.object_detection import draw_bbox

ap = argparse.ArgumentParser()

ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join(["object-coco\coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

print("Loading ...................")
net = cv2.dnn.readNetFromDarknet('object-coco\yolov3.cfg', 'object-coco\yolov3.weights')
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

##vs = cv2.VideoCapture('C:\\Users\\My Lappy\\Desktop\\Object Detection\\videos\\airport.mp4')
vs= cv2.VideoCapture(0)
writer = None
(W, H) = (None, None)

try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1
count=0
sample=0;
error=0
while True:
        ret, img = vs.read()
        print(ret)
        if ret:
                gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cv2.imshow('frame',img)
                cv2.imwrite('frame.png',img)
                        # cv2.waitKey(1)
                sample=sample+1
        if (sample == 20):
                sample =0;
                break
                                
vs.release()
if error ==0:
        print('Camera is interrupted\nPlease execute the script again')
        cv2.destroyAllWindows()
if error ==1:
        print('image is caputured')

        ##        (grabbed, frame) = vs.read()
frame = cv2.imread('frame.png')

# if the frame dimensions are empty, grab them
if W is None or H is None:
        (H, W) = frame.shape[:2]

# construct a blob from the input frame and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes
# and associated probabilities
blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

# initialize our lists of detected bounding boxes, confidences,
# and class IDs, respectively
boxes = []
confidences = []
classIDs = []

# loop over each of the layer outputs
for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > args["confidence"]:
                        # scale the bounding box coordinates back relative to
                        # the size of the image, keeping in mind that YOLO
                        # actually returns the center (x, y)-coordinates of
                        # the bounding box followed by the boxes' width and
                        # height
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        # use the center (x, y)-coordinates to derive the top
                        # and and left corner of the bounding box
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        # update our list of bounding box coordinates,
                        # confidences, and class IDs
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

# apply non-maxima suppression to suppress weak, overlapping
# bounding boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
        args["threshold"])

# ensure at least one detection exists
if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                        confidences[i])
                text1="{}".format(LABELS[classIDs[i]])
                if (text1 == 'car'):
                        bbox, label, conf = cv.detect_common_objects(frame)
                        output_image = draw_bbox(frame, bbox, label, conf)
                        print('Number of cars in the image is '+ str(label.count('car')))
                        break

# release the file pointers
print("[INFO] cleaning up...")
#writer.release()
##vs.release()
