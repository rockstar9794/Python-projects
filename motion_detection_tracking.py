import cv2
import numpy as np

cap = cv2.VideoCapture("video/cars.mp4")
subtractor_MOG = cv2.createBackgroundSubtractorMOG2()
subtractor_KNN = cv2.createBackgroundSubtractorKNN()

history = 200

while(True):
    ret, frame = cap.read()
    if ret is False:
        break
    h,w,d=frame.shape
    orig_frame = frame.copy()
    frame = cv2.resize(frame, (int(w/2), int(h/2)))
    gray =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((5,5), np.uint8)
    mask = subtractor_KNN.apply(frame, learningRate = 2/history) # 0.005
    output = cv2.medianBlur(mask, 17)
    dilation = cv2.dilate(output, kernel, iterations=1)

    contours, hierarcy = cv2.findContours(dilation.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c)<700:
            continue

        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x,y), (w+x, h+y), (0,0,255), 1)

    cv2.imshow('input', frame)
    cv2.imshow('dilate', dilation)
    k= cv2.waitKey(1) &0xff
    if k==ord('q') or k == ord('Q'):
        break
