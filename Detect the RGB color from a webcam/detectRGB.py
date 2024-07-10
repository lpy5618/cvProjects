import cv2
import numpy as np

video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = video.read()
    
    if ret:
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        r=frame[:,:,2:]
        g=frame[:,:,1:2]
        b=frame[:,:,:1]

        r_mean=np.mean(r)
        g_mean=np.mean(g)
        b_mean=np.mean(b)

        if (b_mean>g_mean and b_mean>r_mean):
            print("Blue")
        elif (g_mean>r_mean and g_mean>b_mean):
            print("Green")
        else:
            print("Red")