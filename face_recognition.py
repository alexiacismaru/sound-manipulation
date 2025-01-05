import cv2
import sys

# create a face cascade
cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath) 

# set the video source to the default webcam
video_capture = cv2.VideoCapture(0) 

while True: 
    # Capture the video. The read() function reads one frame from the video source and returns the actual video frame read.
    ret, frame = video_capture.read()

    # search for the face in the captured video
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

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # use 'q' to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()