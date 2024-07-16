import cv2
import numpy as np


# cap = cv2.VideoCapture(0)
# image=cv2.imread("./src/7.png")
def detect_circle(image):
    while True:
        # ret, image = cap.read()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 100, 100])
        upper = np.array([50, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        res = cv2.bitwise_and(image, image, mask=mask)
        ret, thresh1 = cv2.threshold(res, 100, 255, cv2.THRESH_BINARY)
        canny = cv2.Canny(thresh1, 40, 80)
        circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 100, param1=50, param2=30, minRadius=30, maxRadius=150)
        try:

            # 根据检测到圆的信息，画出每一个圆
            for circle in circles[0]:
                if circle[2] >= 100:
                    continue
                # 圆的基本信息
                print(circle[2])
                # 坐标行列
                x = int(circle[0])
                y = int(circle[1])
                # 半径
                r = int(circle[2])

                # 在原图用指定颜色标记出圆的位置
                img = cv2.circle(image, (x, y), r, (0, 0, 255), -1)
        except:
            print("No Circle!")
        cv2.imshow('image', image)
        cv2.imshow('mask', mask)
        cv2.imshow('res', res)
        # cv2.waitKey(0)
        k = cv2.waitKey(5) & 0xff
        if k == 27:
            break


cv2.destroyAllWindows()
