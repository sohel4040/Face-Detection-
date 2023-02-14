# # import tkinter as tk
# from tkinter import *
# def detect():
#     print("Face detection is starting")
# r=Tk()
# r.title("This is a sample frame")
# b=Button(r,text="Detect",command=detect)
# b.pack()
# r.mainloop()

import cv2

cascade_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

    # Capture frame-by-frame
ret, frame = cap.read()
    # Our operations on the frame come here
gray = cv2.cvtColor(frame, 0)

detections = cascade_classifier.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
    
if(len(detections) > 0):
        
    (x,y,w,h) = detections[0]

    frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    # Display the resulting frame
cv2.imshow('frame',frame)

# if cv2.waitKey(1) == ord('q'):
#     break

# When everything done, release the capture
cap.release()

cv2.destroyAllWindows()
