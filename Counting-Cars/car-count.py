from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# For WEBCAM
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap.set(3, 640)
# cap.set(4,480)

cap = cv2.VideoCapture("../Videos/cars.mp4")

model = YOLO("../Yolo-Weights/yolov10n.pt")

mask = cv2.imread("img.png")

#Tracking
tracker = Sort(max_age = 20 , min_hits = 3 , iou_threshold = 0.3)
limits = [400 , 297 , 673 , 297]
totalCounts = []

while True:
    success , img = cap.read()
    imgRegion = cv2.bitwise_and(img , mask)

    result = model(imgRegion , stream = True)
    # print(result)

    #Array to store the Cars bounding box
    detection = np.empty((0,5))

    for r in result :
        boxes = r.boxes
        # print(boxes)
        for box in boxes:
            # Bounding Box
            x1 , y1 , x2 , y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1) , int(y1) , int(x2) , int(y2)
            print(x1 , y1 , x2 , y2)
            # cv2.rectangle(img , (x1 , y1) , (x2 , y2) , (0 , 255 , 0 ) , 3)
            w , h = x2 - x1 , y2 - y1

            #COnfidence
            conf = math.floor((box.conf[0]*100))/100

            #Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == 'car' :
                # cvzone.cornerRect(img, (x1, y1, w, h))
                # cvzone.putTextRect(img , f'{currentClass} {conf}' , (max(0 , x1) , max(30 , y1)) , scale = 0.7 , thickness=1)

                currentArray = np.array([x1,y1,x2,y2,conf])
                detection = np.vstack((detection , currentArray))

    resultTracker = tracker.update(detection)
    cv2.line(img ,(limits[0] , limits[1]) , (limits[2] , limits[3]) , color=(255 , 0 , 0) , thickness=5)
    for result in resultTracker:
        x1 , y1 , x2 , y2 , id = result
        x1, y1, x2, y2 , id = int(x1), int(y1), int(x2), int(y2) , int(id)
        print(result)
        w, h = x2 - x1, y2 - y1

        cvzone.cornerRect(img, (x1, y1, w, h) ,l=9,  rt=5 , colorR=(255,0,0))
        cvzone.putTextRect(img, f'{id}', (max(0, x1), max(30, y1)), scale=0.7, thickness=1)

        cx , cy = x1 + w//2 , y1 + h//2
        cv2.circle(img , (cx,cy) ,5, color=(255,0,255) )

        if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[3] + 20 :
            if totalCounts.count(id) == 0:
                totalCounts.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), color=(0, 255, 0), thickness=5)

    cvzone.putTextRect(img , f'Count : {len(totalCounts)}' , (50 ,50))

    cv2.imshow("Image" , img)
    # cv2.imshow("Mask" , imgRegion)
    cv2.waitKey(0)


