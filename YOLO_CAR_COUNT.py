from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

model = YOLO('../model/yolov8n.pt')

cap = cv2.VideoCapture("car.mp4")
mask = cv2.imread(r'C:\Users\User\OneDrive\Desktop\AI & Data Science\YOLO3\Project-1-Car-Counter\car_edited.png')
KalmanBoxTracker.count = 0
tracker = Sort(max_age=20)
totalcount = []
limit = [166,201,584,201]
pasue= False
while True:
    ret, frame = cap.read()


    if not ret:
        break

    frame_after_edit = cv2.bitwise_and(frame,mask)
    results = model(frame_after_edit,stream=True)
    detections = np.empty((0,5))

    for r in results:
        for box in r.boxes:

            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

            w,h = x2-x1,y2-y1


            conf  = math.ceil((box.conf[0] * 100))/100
            cls = int(box.cls[0])
            currentclass = model.names[cls]
            if (currentclass == 'car' or currentclass == 'truck') and conf > 0.4:
                #cvzone.cornerRect(frame,(x1,y1,w,h),l=9)
                #cvzone.putTextRect(frame,f'{currentclass} {conf}',(max(0,x1),max(35,y1)),scale=0.7,thickness=1)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))


            
    resultTracker = tracker.update(detections)
    cv2.line(frame,(limit[0],limit[1]),(limit[2],limit[3]),color=(0,0,255),thickness=5)
    for result in resultTracker:
        x1,y1,x2,y2,id = result
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        print(result)
        w,h = x2-x1,y2-y1


        #print(result)
        cvzone.cornerRect(frame,(x1,y1,w,h),l=9,rt=2,colorR=(255,0,0))
        cvzone.putTextRect(frame,f'{int(id)}',(max(0,x1),max(35,y1)),scale=0.7,thickness=1)

        cx,cy = x1+w//2,y1+h//2
        cv2.circle(frame,(cx,cy),5,(255,0,255),-1)
        if limit[0] < cx < limit[2] and limit[1]-5 < cy < limit[1]+5:
            if totalcount.count(id) == 0:
                totalcount.append(id)
                cv2.line(frame,(limit[0],limit[1]),(limit[2],limit[3]),color=(0,255,0),thickness=5)


    
    
    cvzone.putTextRect(frame,f'Counter: {len(totalcount)}',(50,50))

    cv2.imshow("Detection", frame)
    cv2.imshow('Mask',frame_after_edit)
    cv2.waitKey(1)
    if cv2.waitKey(2) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()