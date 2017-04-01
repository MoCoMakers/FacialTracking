import cv2
import sys

import socket

import urllib2

from PIL import Image
import io
import httplib

import numpy as np
import urllib

#cascPath = sys.argv[1]
cascPath = "haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_URL="http://10.0.0.43:8080/video"

#video_capture = cv2.VideoCapture('0')

def grabImage4(video_URL):
    stream=urllib.urlopen(video_URL)
    bytes=''
    while True:
        bytes+=stream.read(1024)
        a = bytes.find('\xff\xd8')
        b = bytes.find('\xff\xd9')
        if a!=-1 and b!=-1:
            jpg = bytes[a:b+2]
            bytes= bytes[b+2:]
            i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.CV_LOAD_IMAGE_COLOR)
            #print "Found Image"
	    return i

while True:
    # Capture frame-by-frame
    #ret, frame = video_capture.read()
    image = grabImage4(video_URL)
    frame = image

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
	centerX=x+(w/2)
	centerY=y+(h/2)
	print("X: "+str(centerX)+" Y: "+str(centerY))

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
#video_capture.release()
cv2.destroyAllWindows()

