import cv2
import numpy as np
vid = cv2.VideoCapture(0)

lowH = 0
highH = 10

lowS = 0
highS = 23

lowV = 235
highV = 255

cv2.namedWindow('keypoints')
cv2.namedWindow('threshold')

params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.maxArea = 300
params.minArea = 50

def nothing(x):
    pass

cv2.namedWindow('control')
detector = cv2.SimpleBlobDetector_create(params)
cv2.createTrackbar('lowH', 'control', lowH, 179, nothing)
cv2.createTrackbar('highH', 'control', highH, 179, nothing)

cv2.createTrackbar('lowS','control', lowS, 255, nothing)
cv2.createTrackbar('highS','control', highS, 255, nothing)

cv2.createTrackbar('lowV', 'control',lowV, 255, nothing)
cv2.createTrackbar('highV', 'control', highV, 255, nothing)

while(True):

    lowH = cv2.getTrackbarPos('lowH', 'control')
    highH = cv2.getTrackbarPos('highH', 'control')

    lowS = cv2.getTrackbarPos('lowS', 'control')
    highS = cv2.getTrackbarPos('highS', 'control')

    lowV = cv2.getTrackbarPos('lowV', 'control')
    highV = cv2.getTrackbarPos('highV', 'control')




    # exit simulation when 'q' key is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        flag = True
        break

    ret, view = vid.read()
    hsv_view = cv2.cvtColor(view,cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv_view, np.array([lowH, lowS, lowV]), np.array([highH, highS, highV]))
    cv2.imshow('threshold', thresh)
    thresh = cv2.GaussianBlur(thresh, (5, 5), 0)
    view = cv2.bitwise_and(view,view,mask=thresh)
    keyPoints = detector.detect(view)


    if len(keyPoints) > 0:
        for i in range(0,len(keyPoints)):
            x = '%.1f' % keyPoints[i].pt[0]
            y = '%.1f' % keyPoints[i].pt[1]
            print("Blob detected at (" + str(x)+ " , "+ str(y) + ")")
    im_with_keypoints = cv2.drawKeypoints(view, keyPoints,np.array([]),(255,0,0),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('keypoints', im_with_keypoints)







