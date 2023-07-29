import cv2
import numpy as np

cap = cv2.VideoCapture("LaneVideo.mp4")
success, img = cap.read()

def nothing(x):
    pass

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 200, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)



while success:
    success, img = cap.read()
    frame = cv2.resize(img, (640, 480))

    tl = (222, 387)
    bl = (70, 472)
    tr = (400, 380)
    br = (538, 472)

    cv2.circle(frame, tl, 5, (0, 0, 255), -1)
    cv2.circle(frame, bl, 5, (0, 0, 255), -1)
    cv2.circle(frame, tr, 5, (0, 0, 255), -1)
    cv2.circle(frame, br, 5, (0, 0, 255), -1)

    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_frame = cv2.warpPerspective(frame, matrix, (640, 480))


    hsv_transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv_transformed_frame, lower, upper)




    cv2.imshow("Origninal Video", frame)
    cv2.imshow("Top View", transformed_frame)
    cv2.imshow("Lane Detection-Image Thresholding", mask)

    if cv2.waitKey(1) == 27:
        break
