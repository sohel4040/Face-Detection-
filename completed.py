import cv2
import shutil
import os

cascade_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

def savefaces(frame,faces):
    shutil.rmtree("images")
    os.mkdir("images")

    count=0
    for x,y,w,h in faces:
        # print(x,":",y,":",w,":",h)
        f=frame[(y-20):y+h+20,x-20:x+w+20]
        s=fr"C:\Users\Galaxy\Desktop\Face Detect\images\sample{count}.jpg"
        cv2.imwrite(s,img=f)
        count+=1

count=0

while True:
    
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, 0)

    detections = cascade_classifier.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=2)

    if(len(detections) > 0):
    
        (x,y,w,h) = detections[0]

        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        if cv2.waitKey(1)==ord('c'):
            savefaces(frame,detections)

        # f=frame[(y-20):y+h+20,x-20:x+w+20]
        # s=fr"C:\Users\Galaxy\Desktop\Face Detect\images\sample{count}.jpg"
        # cv2.imwrite(s,img=f)
        # print("Image saved")

        # count+=1

    cv2.imshow('frame',frame)
        
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()