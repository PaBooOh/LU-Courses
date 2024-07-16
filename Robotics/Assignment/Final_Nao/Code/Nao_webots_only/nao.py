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
        self.turnLeft60_1 = Motion('motions/TurnLeft60.motion')
        self.turnLeft40_1 = Motion('motions/TurnLeft40.motion')
        self.turnRight60_1 = Motion('motions/TurnRight60.motion')
        self.turnRight40_1 = Motion('motions/TurnRight40.motion')
        self.turnLeft60_2 = Motion('motions/TurnLeft60.motion')
        self.turnLeft40_2 = Motion('motions/TurnLeft40.motion')
        self.turnRight60_2 = Motion('motions/TurnRight60.motion')
        self.turnRight40_2 = Motion('motions/TurnRight40.motion')
        self.bow = Motion('motions/bow.motion')
        self.pick = Motion('motions/pick.motion')
        self.standup = Motion('motions/standup.motion')

    def get_image_from_camera(self, camera):
        img = camera.getImageArray()
        img = np.asarray(img, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return img

    def detect_by_cameras(self, camera_type='CameraTop'):
        camera = self.getDevice(camera_type)
        detect_time = 0
        l1, u1 = np.array([0, 43, 46]), np.array([10, 255, 255])
        l2, u2 = np.array([156, 43, 46]), np.array([180, 255, 255])
        # l1, u1 = np.array([0,50,50]), np.array([10, 255, 255])
        # l2, u2 = np.array([170,50,50]), np.array([180, 255, 255])
        while robot.step(self.timeStep) != -1:
            detect_time += 1
            if detect_time >= 1000000:
                return False, False
            img = self.get_image_from_camera(camera)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(img_hsv, l1, u1)
            mask2 = cv2.inRange(img_hsv, l2, u2)
            mask3 = mask1 + mask2
            res = cv2.bitwise_and(img, img, mask=mask3)
            _, thresh = cv2.threshold(res, 141, 255, cv2.THRESH_BINARY)
            canny = cv2.Canny(thresh, 0, 0)
            dilate = cv2.dilate(canny, None, iterations=7)
            contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if len(contours) == 0:
                    print('Detecting trash...')
                    self.turnRight60_1.play()
                    detect_time += 1
                    cv2.imshow('Detected', img)
                    cv2.waitKey(1)
                    continue
            cnt_areas = [cv2.contourArea(cnt) for cnt in contours]
            max_cnt_area = max(cnt_areas)
            max_cnt = max(contours, key=lambda cnt: cv2.contourArea(cnt))
            if max_cnt_area <= 400:
                continue
            M = cv2.moments(max_cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(img, "center", (cx - 20, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            # print(f"x: {cx} y: {cy}")
            cv2.drawContours(img, max_cnt, -1, (60, 255, 255), 1)
            cv2.imshow('Detected', img)
            cv2.waitKey(1)
            return cx, cy

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

    def detect_trash_can_bottom(self):
        camera = self.getDevice("CameraBottom")
        hueLow, hueHigh = 0, 11
        saturationLow, saturationHigh = 88, 255
        valueLow, valueHigh = 75, 251
        lower = (hueLow, saturationLow, valueLow)
        upper = (hueHigh, saturationHigh, valueHigh)
        l1, u1 = np.array([0, 43, 46]), np.array([10, 255, 255])
        l2, u2 = np.array([180, 255, 255]), np.array([156, 43, 46])
        while robot.step(self.timeStep) != -1:
            img = self.get_image_from_camera(camera)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(img_hsv, l1, u1)
            mask2 = cv2.inRange(img_hsv, l2, u2)
            mask3 = mask1 + mask2
            res = cv2.bitwise_and(img, img, mask=mask3)
            _, thresh = cv2.threshold(res, 125, 255, cv2.THRESH_BINARY)
            canny = cv2.Canny(thresh, 180, 180)
            dilate = cv2.dilate(canny, None, iterations=1)
            contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if len(contours) == 0:
                return False, False
            for cnt in contours:
                epsilon = 0.1 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                max_cnt_area = max([cv2.contourArea(cnt) for cnt in contours])
                if len(approx) == 4:
                    if cv2.contourArea(cnt) != max_cnt_area:
                        continue
                    M = cv2.moments(cnt)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.circle(img, (cx, cy), 7, (0, 0, 255), -1)
                    cv2.putText(img, "center", (cx - 20, cy - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    print(f"x: {cx} y: {cy}")
                    cv2.drawContours(img, cnt, -1, (60, 255, 255), 2)
                    # cv2.imwrite('OBSTACLES_bt.png', img)
                    cv2.waitKey(1)
                    return cx, cy

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

    def approaching(self, i=0):
        i = i + 2
        while robot.step(self.timeStep) != -1:
            if self.turnRight40_1.isOver() or self.turnLeft40_1.isOver() or self.forwards1.isOver() or self.turnLeft60_1.isOver() or self.turnRight60_1.isOver():
                x, y = self.detect_by_cameras()
                if y >= 218:
                    return True
                print(f"x_{i}: {x} y_{i}: {y}")
                if x > 200:
                    action = 'tl60'
                    print('Left{}'.format(i))
                    self.turnLeft60_1.play()
                    while robot.step(self.timeStep) != -1:
                        if self.turnLeft60_1.isOver():
                            print('Then right{}'.format(i))
                            self.turnRight40_1.play()
                            break
                elif x < 120:
                    action = 'tr60'
                    print('Right{}'.format(i))
                    self.turnRight60_1.play()
                    while robot.step(self.timeStep) != -1:
                        if self.turnRight60_1.isOver():
                            print('And left{}'.format(i))
                            self.turnLeft40_1.play()
                            break
                else:
                    print('Forward{}'.format(i))
                    action = 'fw'
                    self.forwards1.play()
                break
        return False

    def run(self):
        x = 160
        x_bt = 160
        # self.handWave.play()
        # self.setHandsAngle(0.85)
        action = None
        # while robot.step(self.timeStep) != -1:
        #     x, y = self.detect_by_cameras()
        #     print(f"x_1: {x} y_1: {y}")
        while robot.step(self.timeStep) != -1:
            x, y = self.detect_by_cameras()
            # time.sleep(0.5)
            print(f"x: {x} y: {y}")
            if x > 190:
                action = 'tl60'
                print('Left')
                self.turnLeft60_1.play()
                while robot.step(self.timeStep) != -1:
                    if self.turnLeft60_1.isOver():
                        print('And right')
                        self.turnRight40_1.play()
                        break
            elif x < 130:
                action = 'tr60'
                print('Right')
                self.turnRight60_1.play()
                while robot.step(self.timeStep) != -1:
                    if self.turnRight60_1.isOver():
                        print('And left')
                        self.turnLeft40_1.play()
                        break
            else:
                print('Forward1')
                action = 'fw'
                self.forwards1.play()
            break

        dist_flag = False
        time.sleep(5)
        for i in range(6):
            dist_flag = self.approaching(i)
            time.sleep(5)
            if dist_flag:
                print('Close!')
                break



        # while robot.step(self.timeStep) != -1:
        #     if (action == 'tl60' and self.turnLeft60_1.isOver()) or (action == 'tr60' and self.turnRight60_1.isOver()) or (action == 'tl40' and self.turnLeft40_1.isOver()) or (action == 'tr40' and self.turnRight40_1.isOver()) or (action == 'fw' and self.forwards1.isOver()):
        #         x, y = self.detect_by_cameras()
        #         print('Third')
        #         if x > 190:
        #             action = 'tl40'
        #             print('Left3')
        #             self.turnLeft40_1.play()
        #         elif x < 130:
        #             action = 'tr40'
        #             print('Right3')
        #             self.turnRight40_1.play()
        #         else:
        #             print('Forward3')
        #             action = 'fw'
        #             self.forwards1.play()


        # while robot.step(self.timeStep) != -1:
        #     x, y = self.detect_by_cameras()
        #     # time.sleep(0.5)
        #     if x > 190:
        #         action = 'tl40'
        #         print('Left')
        #         self.turnLeft40_1.play()
        #     elif x < 130:
        #         action = 'tr40'
        #         print('Right')
        #         self.turnRight40_1.play()
        #     else:
        #         self.forwards1.play()
        #     break

        key = -1
        while robot.step(self.timeStep) != -1:
            if self.handWave.isOver():
                time.sleep(1)
                self.bow.play()
                break

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
            if self.standup.isOver():
                x,y = self.detect_by_cameras()
                break

        while robot.step(self.timeStep) != -1:
            if x != 160:
                x_flag = x
                time.sleep(0.5)
                if x > 190:
                    print('Left')
                    self.turnLeft60_1.play()
                elif x < 130:
                    print('Right')
                    self.turnRight60_1.play()
                else:
                    self.forwards1.play()
                break

        while robot.step(self.timeStep) != -1:
            if x != 160:
                x_flag = x
                time.sleep(0.5)
                if x > 190:
                    print('Left')
                    self.turnLeft60_1.play()
                elif x < 130:
                    print('Right')
                    self.turnRight60_1.play()
                else:
                    self.forwards1.play()
                break

        while robot.step(self.timeStep) != -1:
            if self.turnLeft60_1.isOver():
                time.sleep(0.5)
                self.turnRight40_1.play()
                break
            elif self.turnRight60_1.isOver():
                time.sleep(0.5)
                self.turnLeft40_1.play()
                break

        while robot.step(self.timeStep) != -1:
            if self.turnRight40_1.isOver() or self.turnLeft40_1.isOver():
                time.sleep(0.5)
                self.forwards1.play()
                break

        while robot.step(self.timeStep) != -1:
            if self.forwards1.isOver():
                self.forwards2.play()
                break

        while robot.step(self.timeStep) != -1:
            if self.forwards2.isOver():
                x_bt, y_bt = self.detect_trash_can_bottom()
                break

        while robot.step(self.timeStep) != -1:
            if x_bt != 160:
                time.sleep(0.5)
                # print(1)
                if x < 120:
                    self.turnLeft60_2.play()
                elif x > 180:
                    self.turnRight60_2.play()
                else:
                    # print(2)
                    self.forwards.play()
                break
        while robot.step(self.timeStep) != -1:
            if self.turnLeft60_2.isOver():
                time.sleep(0.5)
                self.turnRight40_2.play()
                break
            elif self.turnRight60_2.isOver():
                time.sleep(0.5)
                self.turnLeft40_2.play()
                break
        while robot.step(self.timeStep) != -1:
            if self.turnRight40_2.isOver() or self.turnLeft40_2.isOver():
                time.sleep(0.5)
                self.forwards.play()
                break
        while robot.step(self.timeStep) != -1:
            if self.forwards.isOver():
                time.sleep(0.5)
                self.backwards.play()
                break
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
