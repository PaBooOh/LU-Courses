import time

from controller import Robot, Keyboard, Motion, Camera, CameraRecognitionObject
import cv2
import numpy as np


class Nao(Robot):
    PHALANX_MAX = 8

    # load motion files
    def loadMotionFiles(self):
        self.handWave = Motion('motions/HandWave.motion')
        self.forwards = Motion('motions/Forwards.motion')
        self.forwards1 = Motion('motions/Forwards50.motion')
        self.forwards2 = Motion('motions/Forwards50.motion')
        self.backwards = Motion('motions/Backwards.motion')
        self.sideStepLeft = Motion('motions/SideStepLeft.motion')
        self.sideStepRight = Motion('motions/SideStepRight.motion')
        self.turnLeft60 = Motion('motions/TurnLeft60.motion')
        self.turnLeft40 = Motion('motions/TurnLeft40.motion')
        self.turnRight60 = Motion('motions/TurnRight60.motion')
        self.turnRight40 = Motion('motions/TurnRight40.motion')
        self.shoot = Motion('motions/Shoot.motion')
        self.neckDetect = Motion('motions/NeckDetection.motion')
        self.bow = Motion('motions/bow.motion')
        self.pick = Motion('motions/pick.motion')
        self.standup = Motion('motions/standup.motion')

    def startMotion(self, motion):
        # interrupt current motion
        if self.currentlyPlaying:
            self.currentlyPlaying.stop()

        # start new motion
        motion.play()
        self.currentlyPlaying = motion

    def dectect_by_neck(self):
        self.neckDetect.play()

    def get_image_from_camera(self, camera):
        img = camera.getImageArray()
        img = np.asarray(img, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return img

    def setHandsAngle(self, angle):
        for i in range(0, self.PHALANX_MAX):
            clampedAngle = angle
            if clampedAngle > self.maxPhalanxMotorPosition[i]:
                clampedAngle = self.maxPhalanxMotorPosition[i]
            elif clampedAngle < self.minPhalanxMotorPosition[i]:
                clampedAngle = self.minPhalanxMotorPosition[i]

            if len(self.rphalanx) > i and self.rphalanx[i] is not None:
                self.rphalanx[i].setPosition(clampedAngle)
            if len(self.lphalanx) > i and self.lphalanx[i] is not None:
                self.lphalanx[i].setPosition(clampedAngle)

    def detect_trash_can(self):
        detect_time = 0
        camera = self.getDevice("CameraTop")
        hueLow, hueHigh = 0, 11
        saturationLow, saturationHigh = 88, 255
        valueLow, valueHigh = 75, 251
        lower = (hueLow, saturationLow, valueLow)
        upper = (hueHigh, saturationHigh, valueHigh)
        l1, u1 = np.array([0,43,46]), np.array([10,255,255])
        l2, u2 = np.array([180, 255, 255]), np.array([156, 43, 46])
        while robot.step(self.timeStep) != -1:
            self.forwards.play()
            img = self.get_image_from_camera(camera)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(img_hsv, l1, u1)
            mask2 = cv2.inRange(img_hsv, l2, u2)
            # mask2 = cv2.inRange(img_hsv, (175, 50, 20), (180, 255, 255))
            # mask = cv2.bitwise_or(mask1, mask2)
            # mask = cv2.inRange(img_hsv, lower, upper)
            # mask = cv2.GaussianBlur(mask, (5, 5), 0)
            mask3 = mask1 + mask2
            res = cv2.bitwise_and(img, img, mask=mask3)
            _, thresh = cv2.threshold(res, 125, 255, cv2.THRESH_BINARY)
            canny = cv2.Canny(thresh, 180, 180)
            dilate = cv2.dilate(canny, None, iterations=1)
            contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            # if len(contours) == 0:
            #     if detect_time == 2:
            #         return False, False
            #     else:
            #         print('Detecting trash...')
            #         self.turnRight60.play()
            #         detect_time += 1
            hit = False
            for cnt in contours:
                epsilon = 0.1 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                max_cnt_area = max([cv2.contourArea(cnt) for cnt in contours])
                if len(approx) == 4:
                    if cv2.contourArea(cnt) != max_cnt_area:
                        continue
                    if max_cnt_area <= 700:
                        continue
                    hit = True
                    M = cv2.moments(cnt)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.circle(img, (cx, cy), 7, (0, 0, 255), -1)
                    cv2.putText(img, "center", (cx - 20, cy - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                    print(f"x: {cx} y: {cy}")
                    cv2.drawContours(img, cnt, -1, (60, 255, 255), 2)
                    cv2.imshow('OBSTACLES', img)
                    cv2.waitKey(1)
                    # return cx, cy
            # cv2.imshow('OBSTACLES', img)

            # if not hit:
            #     print('Turn right...')
            #     self.turnRight40.play()




    def findAndEnableDevices(self):
        # get the time step of the current world.
        self.timeStep = int(self.getBasicTimeStep())

        # camera
        self.cameraTop = self.getDevice("CameraTop")
        self.cameraBottom = self.getDevice("CameraBottom")
        self.cameraTop.enable(4 * self.timeStep)
        self.cameraBottom.enable(4 * self.timeStep)

        # accelerometer
        self.accelerometer = self.getDevice('accelerometer')
        self.accelerometer.enable(4 * self.timeStep)

        # gyro
        self.gyro = self.getDevice('gyro')
        self.gyro.enable(4 * self.timeStep)

        # gps
        self.gps = self.getDevice('gps')
        self.gps.enable(4 * self.timeStep)

        # inertial unit
        self.inertialUnit = self.getDevice('inertial unit')
        self.inertialUnit.enable(self.timeStep)

        # ultrasound sensors
        self.us = []
        usNames = ['Sonar/Left', 'Sonar/Right']
        for i in range(0, len(usNames)):
            self.us.append(self.getDevice(usNames[i]))
            self.us[i].enable(self.timeStep)

        # foot sensors
        self.fsr = []
        fsrNames = ['LFsr', 'RFsr']
        for i in range(0, len(fsrNames)):
            self.fsr.append(self.getDevice(fsrNames[i]))
            self.fsr[i].enable(self.timeStep)

        # foot bumpers
        self.lfootlbumper = self.getDevice('LFoot/Bumper/Left')
        self.lfootrbumper = self.getDevice('LFoot/Bumper/Right')
        self.rfootlbumper = self.getDevice('RFoot/Bumper/Left')
        self.rfootrbumper = self.getDevice('RFoot/Bumper/Right')
        self.lfootlbumper.enable(self.timeStep)
        self.lfootrbumper.enable(self.timeStep)
        self.rfootlbumper.enable(self.timeStep)
        self.rfootrbumper.enable(self.timeStep)

        # there are 7 controlable LED groups in Webots
        self.leds = []
        self.leds.append(self.getDevice('ChestBoard/Led'))
        self.leds.append(self.getDevice('RFoot/Led'))
        self.leds.append(self.getDevice('LFoot/Led'))
        self.leds.append(self.getDevice('Face/Led/Right'))
        self.leds.append(self.getDevice('Face/Led/Left'))
        self.leds.append(self.getDevice('Ears/Led/Right'))
        self.leds.append(self.getDevice('Ears/Led/Left'))

        # get phalanx motor tags
        # the real Nao has only 2 motors for RHand/LHand
        # but in Webots we must implement RHand/LHand with 2x8 motors
        self.lphalanx = []
        self.rphalanx = []
        self.maxPhalanxMotorPosition = []
        self.minPhalanxMotorPosition = []
        for i in range(0, self.PHALANX_MAX):
            self.lphalanx.append(self.getDevice("LPhalanx%d" % (i + 1)))
            self.rphalanx.append(self.getDevice("RPhalanx%d" % (i + 1)))

            # assume right and left hands have the same motor position bounds
            self.maxPhalanxMotorPosition.append(self.rphalanx[i].getMaxPosition())
            self.minPhalanxMotorPosition.append(self.rphalanx[i].getMinPosition())

        # shoulder pitch motors
        self.RShoulderPitch = self.getDevice("RShoulderPitch")
        self.LShoulderPitch = self.getDevice("LShoulderPitch")

        # keyboard
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(10 * self.timeStep)

    def __init__(self):
        Robot.__init__(self)
        self.currentlyPlaying = False

        # initialize stuff
        self.findAndEnableDevices()
        self.loadMotionFiles()

    def run(self):
        x = 160
        # flag1 = True
        # flag2 = True
        self.bow.play()
        self.setHandsAngle(0.85)
        # until a key is pressed
        key = -1
        # self.forwards.setLoop(True)
        while robot.step(self.timeStep) != -1:
            if self.bow.isOver():
                time.sleep(1)
                self.pick.play()
                break

        while robot.step(self.timeStep) != -1:
            if self.pick.isOver():
                time.sleep(0.5)
                self.standup.play()
                break

        while robot.step(self.timeStep) != -1:
            # for i in range(10):
            if self.standup.isOver():
                break

        x, y = self.detect_trash_can()
        if x and y:
            print(f"x: {x} y: {y}")
        # while robot.step(self.timeStep) != -1:
        #     if x != 160:
        #         time.sleep(0.5)
        #         if x < 130:
        #             self.turnLeft60.play()
        #         elif x > 190:
        #             self.turnRight60.play()
        #         break

        # if x < 130:
        #     while robot.step(self.timeStep) != -1:
        #         if self.standup.isOver():
        #             time.sleep(0.5)
        #             self.turnLeft60_1.play()
        #             break
        #
        #     while robot.step(self.timeStep) != -1:
        #         if self.turnLeft60_1.isOver():
        #             time.sleep(0.5)
        #             self.turnRight40_1.play()
        #             break
        # elif x > 190:
        #     while robot.step(self.timeStep) != -1:
        #         if self.standup.isOver():
        #             time.sleep(0.5)
        #             self.turnRight60_1.play()
        #             break
        #
        #     while robot.step(self.timeStep) != -1:
        #         if self.turnRight60_1.isOver():
        #             time.sleep(0.5)
        #             self.turnLeft40_1.play()
        #             break

        # while robot.step(self.timeStep) != -1:
        #     if self.turnRight40_1.isOver() or self.turnLeft40_1.isOver():
        #         time.sleep(0.5)
        #         self.forwards1.play()
        #         break
        #
        # while flag2:
        #     if self.forwards1.isOver():
        #         x,y = self.detect_trash_can()
        #         flag2 = False
        #     break
        #
        # if x < 130:
        #     while robot.step(self.timeStep) != -1:
        #         if self.forwards1.isOver():
        #             time.sleep(0.5)
        #             self.turnLeft60_2.play()
        #             break
        #
        #     while robot.step(self.timeStep) != -1:
        #         if self.turnLeft60_2.isOver():
        #             time.sleep(0.5)
        #             self.turnRight40_2.play()
        #             break
        # elif x > 190:
        #     while robot.step(self.timeStep) != -1:
        #         if self.forwards1.isOver():
        #             time.sleep(0.5)
        #             self.turnRight60_2.play()
        #             break
        #
        #     while robot.step(self.timeStep) != -1:
        #         if self.turnRight60_2.isOver():
        #             time.sleep(0.5)
        #             self.turnLeft40_2.play()
        #             break
        #
        # while robot.step(self.timeStep) != -1:
        #     if self.turnRight40_2.isOver() or self.turnLeft40_2.isOver():
        #         time.sleep(0.5)
        #         self.forwards2.play()
        #         break

        # while robot.step(self.timeStep) != -1:
        #     if self.forwards2.isOver():
        #         time.sleep(0.5)
        #         self.forwards.play()
        #         break
        #
        # while robot.step(self.timeStep) != -1:
        #     if self.forwards.isOver():
        #         time.sleep(0.5)
        #         self.backwards.play()
        #         break


        while robot.step(self.timeStep) != -1:
            key = self.keyboard.getKey()
            if key > 0:
                break

        # self.forwards.setLoop(False)


        while True:
            key = self.keyboard.getKey()

            if key == ord('Q'):
                break

            if robot.step(self.timeStep) == -1:
                break


# create the Robot instance and run main loop
robot = Nao()
robot.run()
