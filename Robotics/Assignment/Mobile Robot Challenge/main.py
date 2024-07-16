import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while(1):
    # get a frame
    ret, frame = cap.read()
    # show a frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 输出图像大小，方便根据图像大小调节minRadius和maxRadius
    print(frame.shape)
    # cv2.imshow('gray',gray)

    # th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    #                             cv2.THRESH_BINARY, 11, 2)
    # cv2.imshow('binary', th2)

    mb = cv2.medianBlur(frame, 3)
    # cv2.imshow('mb', mb)

    ret, thresh1 = cv2.threshold(mb, 200, 255, cv2.THRESH_BINARY)
    # cv2.imshow('thresh1', thresh1)

    canny = cv2.Canny(thresh1, 40, 80)
    # cv2.imshow('Canny', canny)

    canny = cv2.blur(canny, (3, 3))
    # cv2.imshow('blur', canny)

    # 霍夫变换圆检测
    circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 100, param1=50, param2=30, minRadius=30, maxRadius=150)
    # 输出返回值，方便查看类型
    # print(circles)
    try:

        # 根据检测到圆的信息，画出每一个圆
        for circle in circles[0]:
            if (circle[2] >= 100):
                continue
            # 圆的基本信息
            print(circle[2])
            # 坐标行列
            x = int(circle[0])
            y = int(circle[1])
            # 半径
            r = int(circle[2])

            # 在原图用指定颜色标记出圆的位置
            img = cv2.circle(frame, (x, y), r, (0, 0, 255), -1)
    except:
        print("No Circle!")

    cv2.imshow("capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break