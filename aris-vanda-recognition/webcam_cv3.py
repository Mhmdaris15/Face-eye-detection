import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    if len(faces) == 2:
        print(faces)
        # for (x, y, w, h) in faces[1]:
        #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #     cv2.putText(frame, f'Muhammad Aris\n{len(faces)=}', (x,y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,250), 2, cv2.LINE_AA)

    # Draw a rectangle around the faces
    for i, coord in enumerate(faces, start=1):
        if i == 2:
            cv2.rectangle(frame, (coord[0], coord[1]), (coord[0]+coord[2], coord[1]+coord[3]), (0, 255, 0), 2)
            cv2.putText(frame, f'Vanda Margraf', (coord[0],coord[1]+coord[3]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,250), 2, cv2.LINE_AA)
            continue
        cv2.rectangle(frame, (coord[0], coord[1]), (coord[0]+coord[2], coord[1]+coord[3]), (0, 255, 0), 2)
        cv2.putText(frame, f'Muhammad Aris', (coord[0],coord[1]+coord[3]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,250), 2, cv2.LINE_AA)

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
