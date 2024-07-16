# -*- coding: utf-8 -*-
import time, math
import numpy as np
import naoqi
import qi
import cv2
from naoqi import ALProxy
from PIL import Image

class Nao():
    def __init__(self):
        self.IP = "127.0.0.1"
        self.PORT = 12344
        self.motion = ALProxy('ALMotion', self.IP, self.PORT)
        self.posture = ALProxy('ALRobotPosture', self.IP, self.PORT)
        self.tracker = ALProxy('ALTracker', self.IP, self.PORT)
        self.video = ALProxy('ALVideoDevice', self.IP, self.PORT)
        self.tts = ALProxy('ALTextToSpeech', self.IP, self.PORT)
        # self.landmark = ALProxy('ALLandMarkDetection', IP, PORT)
        self.memory = ALProxy('ALMemory', self.IP, self.PORT)
        self.resolution = 2
        self.colorSpace = 11
        self.angleSearch = 60 * math.pi / 180
        self.fps = 5
    
    def detect(self):
        i = 1
        maxAngleScan = self.angleSearch
        while True:
            if i % 8 == 1:
                self.motion.angleInterpolationWithSpeed("Head", [-maxAngleScan, 0.035], 0.1)
            elif i % 8 == 2:
                self.motion.angleInterpolationWithSpeed("Head", [-2 * maxAngleScan / 3, 0.035], 0.1)
            elif i % 8 == 3:
                self.motion.angleInterpolationWithSpeed("Head", [-maxAngleScan / 3, 0.035], 0.1)
            elif i % 8 == 4:
                self.motion.angleInterpolationWithSpeed("Head", [0, 0.035], 0.1)
            elif i % 8 == 5:
                self.motion.angleInterpolationWithSpeed("Head", [maxAngleScan / 3, 0.035], 0.1)
            elif i % 8 == 6:
                self.motion.angleInterpolationWithSpeed("Head", [2 * maxAngleScan / 3, 0.035], 0.1)
            elif i % 8 == 7:
                self.motion.angleInterpolationWithSpeed("Head", [maxAngleScan, 0.035], 0.1)
            self.get_image_from_camera_top()
            # self.display_image(img)
            i += 1
            if i == 1000:
                break
    
    def display_image(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 0])
        upper = np.array([50, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        res = cv2.bitwise_and(img, img, mask=mask)
        canny = cv2.Canny(res, 60, 80)
        circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 100, param1=50, param2=30, minRadius=1,
                                    maxRadius=20)
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
                img = cv2.circle(img, (x, y), r, (0, 0, 255), -1)
        except:
            print("No Circle!")
        cv2.imshow('image', img)
        cv2.imshow('mask', mask)
        cv2.imshow('canny', canny)
        cv2.imshow('res', res)
        cv2.waitKey(10) & 0xff

    def get_image_from_camera_top(self, camera_index=0):
        # Subscribes to ALVideoDevice as VideoDeviceProxyvideo.
        VideoDeviceProxy = self.video.subscribeCamera("python_client", camera_index, self.resolution, self.colorSpace, self.fps)
        # Retrieves the latest image from the video source.
        latestImage = self.video.getImageRemote(VideoDeviceProxy)
        # Unregisters a module from ALVideoDevice.
        self.video.unsubscribe(VideoDeviceProxy)
        # Image size
        imageWidth = latestImage[0]
        imageHeight = latestImage[1]
        # array contains the image data passed as an array of ASCII chars.
        array = latestImage[6]
        # Create a PIL Image from the pixel array.
        img = Image.frombytes("RGB", (imageWidth, imageHeight), array)
        # Save the image.
        img = np.asarray(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('img', img)
        cv2.waitKey(5) & 0xff
        # return img

    def get_image_from_camera_bottom(self):
        pass

    def rotate_robot_head(self):
        # Interpolates joints associated with "Head" to a target angle ([0, 0.035]), using a fraction of max speed.
        self.motion.angleInterpolationWithSpeed("Head", [0, 0.035], 0.1)


nao = Nao()
nao.detect()