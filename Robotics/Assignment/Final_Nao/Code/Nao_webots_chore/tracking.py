import qi, time, math
from naoqi import ALProxy
import cv2
from PIL import Image
import numpy as np

# Robot configuration.
IP = "127.0.0.1"
PORT = 12344
motion = ALProxy('ALMotion', IP, PORT)
posture = ALProxy('ALRobotPosture', IP, PORT)
tracker = ALProxy('ALTracker', IP, PORT)
video = ALProxy('ALVideoDevice', IP, PORT)
tts = ALProxy('ALTextToSpeech', IP, PORT)
# landmark = ALProxy('ALLandMarkDetection', robotIP, 13333)
memory = ALProxy('ALMemory', IP, PORT)

# Resolution 640x480px - VGA
resolution = 2  # for 1280x960 change to 3.
# Colorspace BBGGRR - RGB
colorSpace = 11
# frame rate.
fps = 5


# This function activates on nao .wakeUp(). It looks for the ball.
def first_scan():
    # Convert 60 deg to radians.
    angleSearch = 60 * math.pi / 180
    ang = 100
    # Initializes the move process. Checks the robot pose and takes a right posture.
    motion.moveInit()

    # There are two identical camera on the forehead.
    cameraIndexValue = 0  # Activate the top camera. 1 for lower.

    state = 0  # Pointer to check if ball was detected else movement is required.
    while ang == 100:
        [centerOfMassValue, motionAnglesValue] = ball_scan(angleSearch, cameraIndexValue)
        # Check if ball was detected.
        ang, found = rotate_center_head(centerOfMassValue, motionAnglesValue)
        # If ball was not detected using bottom camera, activate top camera and move.
        if found == 0 and cameraIndexValue == 1:
            state = state + 1
            cameraIndexValue = 0
            # moveTo(x,y, rotation) makes NAO move to the given pose in the ground plane, relative to FRAME_ROBOT.
            motion.moveTo(0, 0, 2 * math.pi / 3)
        # if the ball was not detected using top camera, activate bottom camera.
        elif found == 0 and cameraIndexValue == 0:
            cameraIndexValue = 1
        # if state reaches value of 3, make nao move.
        if state == 3:
            # Make nao speak.
            tts.say('I need to move to find the ball')
            # moveTo(x,y, rotation) makes NAO move to the given pose in the ground plane, relative to FRAME_ROBOT.
            motion.moveTo(0.3, 0, 0)
            state = 0

    # when ang is not 100.
    else:
        # moveTo(x,y, rotation) makes NAO move to the given pose in the ground plane, relative to FRAME_ROBOT.
        motion.moveTo(0, 0, ang * 7 / 6)

    getImage("couldBeBall.png", 0)

    # Analyse position of ball.
    [centerOfMassValue1, motionAnglesValue1] = ball_location(math.pi / 9, cameraIndexValue)

    [ang, X, delta] = findBallLocation(centerOfMassValue1, motionAnglesValue1)
    if ang == 100:
        # block cameraIndexValue so it does not activate other if loops.
        cameraIndexValue = 2

    # moveTo(x,y, rotation) makes NAO move to the given pose in the ground plane, relative to FRAME_ROBOT.
    motion.moveTo(0, 0, ang * 7 / 6)

    # Read img and get ball's center of mass.
    img = cv2.imread("couldBeBall.png")
    centerOfMass = CenterOfMassUp(img)

    return centerOfMass, delta, cameraIndexValue


# This function is used to check if the ball is in front of Nao robot.
def ball_scan(angleSearch, CameraIndex):
    # Defining the area of search.
    # Angle of rotation is [-maxAngleScan;+maxAngleScan].
    names = "HeadYaw"
    useSensors = False
    motionAngles = []
    maxAngleScan = angleSearch

    i = 1
    while i < 8:
        # Interpolates joints associated with "Head" to a target angle ([-maxAngleScan, 0.035]), using a fraction of max speed.
        if i == 1:
            motion.angleInterpolationWithSpeed("Head", [-maxAngleScan, 0.035], 0.1)
        elif i == 2:
            motion.angleInterpolationWithSpeed("Head", [-2 * maxAngleScan / 3, 0.035], 0.1)
        elif i == 3:
            motion.angleInterpolationWithSpeed("Head", [-maxAngleScan / 3, 0.035], 0.1)
        elif i == 4:
            motion.angleInterpolationWithSpeed("Head", [0, 0.035], 0.1)
        elif i == 5:
            motion.angleInterpolationWithSpeed("Head", [maxAngleScan / 3, 0.035], 0.1)
        elif i == 6:
            motion.angleInterpolationWithSpeed("Head", [2 * maxAngleScan / 3, 0.035], 0.1)
        elif i == 7:
            motion.angleInterpolationWithSpeed("Head", [maxAngleScan, 0.035], 0.1)

        # Retrieve image from current angle and saves it.
        getImage('getImage' + str(i) + '.png', CameraIndex)
        # Retrives the angles of joints.
        commandAngles = motion.getAngles(names, useSensors)
        # Append retrieved commandAngles into motionAngles.
        motionAngles.append(commandAngles)

        i = i + 1

    # Find center of the detected ball in each image.
    x = i
    centers = findCenter(x)

    # return found centers and array of all angles of joints.
    return [centers, motionAngles]


# Find the ball and center its look to it, otherwise back to 0 and rotate again
def findBallLocation(centers, rot_angles):
    index = centerOfMassBall(centers)
    if len(index) == 0:
        string = "Where is the ball?"
        ang = 100
        state = 0
        RF = 0
    elif len(index) == 1:
        a = index[0]
        string = "I need to get closer to the ball."
        ang = rot_angles[a][0]
        # ang = ang.item()
        state = 1
        RF = 0
        motion.angleInterpolationWithSpeed("Head", [ang, 0.035], 0.1)
    else:
        string = "I see the ball."
        a = index[0]
        b = index[1]
        RF = (rot_angles[b][0] - rot_angles[a][0]) / (centers[a][1] - centers[b][1])
        ang = rot_angles[a][0] - (320 - centers[a][1]) * RF
        # ang = ang.item()
        state = 2
        motion.angleInterpolationWithSpeed("Head", [ang, 0.035], 0.1)
    tts.say(string)

    return [ang, state, RF]


# Retrieves 1 image from Nao using the appropriate camera index - 0 for top, 1 for bottom.
def getImage(name, CameraIndex):
    # Subscribes to ALVideoDevice as VideoDeviceProxyvideo.
    VideoDeviceProxy = video.subscribeCamera("python_client", CameraIndex, resolution, colorSpace, fps)
    # Retrieves the latest image from the video source.
    latestImage = video.getImageRemote(VideoDeviceProxy)
    # Unregisters a module from ALVideoDevice.
    video.unsubscribe(VideoDeviceProxy)

    # Image size
    imageWidth = latestImage[0]
    imageHeight = latestImage[1]
    # array contains the image data passed as an array of ASCII chars.
    array = latestImage[6]
    # Create a PIL Image from the pixel array.
    im = Image.frombytes("RGB", (imageWidth, imageHeight), str(array))
    # Save the image.
    im.save(name, "png")


# Retrieves multiple images to find location of ball.
def ball_location(angleScan, CameraIndex):
    # Defining the area of search.
    # Angle of rotation is [-maxAngleScan;+maxAngleScan].
    names = "HeadYaw"
    useSensors = False
    motionAngles = []
    maxAngleScan = angleScan
    i = 1
    while i < 4:
        if i == 1:
            # Interpolates joints associated with "Head" to a target angle ([-maxAngleScan, 0.035]), using a fraction of max speed.
            motion.angleInterpolationWithSpeed("Head", [-maxAngleScan, 0.035], 0.1)
        elif i == 2:
            # Interpolates joints associated with "Head" to a target angle ([-maxAngleScan, 0.035]), using a fraction of max speed.
            motion.angleInterpolationWithSpeed("Head", [0, 0.035], 0.1)
        elif i == 3:
            # Interpolates joints associated with "Head" to a target angle ([-maxAngleScan, 0.035]), using a fraction of max speed.
            motion.angleInterpolationWithSpeed("Head", [maxAngleScan, 0.035], 0.1)

        # Retrieve image from current angle and saves it.
        getImage('ballDetected' + str(i) + '.png', CameraIndex)
        # Retrives the angles of joints.
        commandAngles = motion.getAngles(names, useSensors)
        # Append retrieved commandAngles into motionAngles.
        motionAngles.append(commandAngles)

        i = i + 1

    x = i
    # find center of ball and return it as an array.
    centers = findCenter(x)

    return [centers, motionAngles]


# Find center of ball.
def findCenter(x):
    centerOfMass = []
    if x == 8:
        for i in range(1, x):
            img = cv2.imread("getImage" + str(i) + ".png")
            cm = CenterOfMassUp(img)
            centerOfMass.append(cm)
    elif x == 4:
        for i in range(1, x):
            img = cv2.imread("ballDetected" + str(i) + ".png")
            cm = CenterOfMassUp(img)
            centerOfMass.append(cm)

    return centerOfMass


# Check if the ball was detected. If it was, it would have a center of mass.
def centerOfMassBall(centerOfMass):
    centerOfMassBallArray = []
    for i in range(len(centerOfMass)):
        if centerOfMass[i] != [0, 0]:
            centerOfMassBallArray.append(i)
    return centerOfMassBallArray


# Center head to ball's center, else rotate again.
def rotate_center_head(centers, rot_angles):
    index = centerOfMassBall(centers)
    found = 1
    # len(index) is 0 if ball isn't detected.
    if len(index) == 0:
        string = "I cannot see the ball."
        ang = 100
        found = 0
    # len(index) is 1 if ball is in Nao's vision.
    elif len(index) == 1:
        a = index[0]
        string = "The ball is there. I will need to get closer."
        ang = rot_angles[a][0]
    else:
        string = "Here is the ball."
        a = index[0]
        b = index[1]
        den = 3
        if len(index) < 3:
            ang = (rot_angles[b][0] + rot_angles[a][0]) / 2
        else:
            c = index[2]
            ang = (rot_angles[b][0] + rot_angles[a][0] + rot_angles[c][0]) / 3

    # Interpolates joints associated with "Head" to a target angle ([0, 0.035]), using a fraction of max speed.
    motion.angleInterpolationWithSpeed("Head", [0, 0.035], 0.1)
    # Make Nao speak the string.
    tts.say(string)

    return ang, found


# Calculate CoM of the thresholded ball.
def CenterOfMassUp(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lowera = np.array([1, 190, 200])
    uppera = np.array([18, 255, 255])
    lowerb = np.array([1, 190, 200])
    upperb = np.array([18, 255, 255])

    mask1 = cv2.inRange(hsv, lowera, uppera)
    mask2 = cv2.inRange(hsv, lowerb, upperb)
    mask = cv2.add(mask1, mask2)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cont, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if len(cont) >= 1:
        minim = 0
        minE = 1
        for l in range(len(cont)):
            contour = cont[l]
            center, radius = cv2.minEnclosingCircle(contour)
            icenter = []
            icenter.append(int(center[0]))
            icenter.append(int(center[1]))
            radius = int(radius)
            d = radius
            err = []
            for k in range(len(contour)):
                t = [contour[k, 0, 0] - center[0], contour[k, 0, 1] - center[1]]
                t = math.sqrt(math.pow(t[0], 2) + math.pow(t[1], 2))
                e = abs(d - t) / d
                err = err + [e]
            ERR = np.mean(err)
            if ERR < minE and radius >= 8 and radius < 65:
                minim = l
                minE = ERR
                Radius = radius
                Center = icenter
        if minE > .235:
            i = 0
            j = 0
            contour = []
            Radius = 0
        else:
            i = Center[1]
            j = Center[0]
            contour = cont[minim]
    else:
        i = 0
        j = 0
        minim = 1
        Center = [1, 1]
        contour = cont
        Radius = 0

    centerOfMass = [i, j]

    return centerOfMass


def CenterOfMassDown(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)
    lowera = np.array([1, 190, 200])
    uppera = np.array([18, 255, 255])
    lowerb = np.array([1, 190, 200])
    upperb = np.array([18, 255, 255])
    mask1 = cv2.inRange(hsv, lowera, uppera)
    mask2 = cv2.inRange(hsv, lowerb, upperb)
    mask = cv2.add(mask1, mask2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    _, cont, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    icenter = []
    if len(cont) >= 1:
        maxim = 0
        for i in range(len(cont)):
            if len(cont[i]) > len(cont[maxim]):
                maxim = i
        contour = cont[maxim]

        center, radius = cv2.minEnclosingCircle(contour)
        icenter.append(int(center[0]))
        icenter.append(int(center[1]))
        radius = int(radius)
        if radius > 8 and radius < 60:
            i = icenter[1]
            j = icenter[0]
        else:
            i = 0
            j = 0
    else:
        i = 0
        j = 0
        contour = cont
    centerOfMass = [i, j]
    return centerOfMass


def zero_head():
    motion.angleInterpolationWithSpeed("HeadYaw", 0, 0.1)


def walkDown(cm, delta):
    idx = 1
    pp = "ball_downfront"
    ext = ".png"
    # motion.moveTo(0.2, 0, alpha*7/6)
    motion.moveTo(0.2, 0, 0)
    while cm[0] > 0 and cm[0] < 230:
        # motion.moveTo(0.2, 0, 0)
        im_num = pp + str(idx) + ext
        getImage(im_num, 1)
        img = cv2.imread(im_num)
        cm = CenterOfMassUp(img)
        if cm == [0, 0]:
            return 0, cm
        alpha = (cm[1] - 320) * delta
        motion.moveTo(0.2, 0, alpha * 7 / 6)
        idx = idx + 1
    # Tilt the head so it can have a better look of the ball
    anglePitch = math.pi * 20.6 / 180
    motion.angleInterpolationWithSpeed("HeadPitch", anglePitch, 0.1)
    # The threshold of 300 is equal to a distance of 15cm from the ball
    # The robot will do a small walk of 7cm and exit the loop
    while cm[0] >= 0 and cm[0] < 300:
        im_num = pp + str(idx) + ext
        getImage(im_num, 1)
        img = cv2.imread(im_num)
        cm = CenterOfMassDown(img)
        if cm == [0, 0]:
            return 0, cm
        if cm[0] < 350:
            alpha = (cm[1] - 320) * delta
            motion.moveTo(0.07, 0, alpha * 8 / 6)
        else:
            break
        idx = idx + 1
    taskComplete = 1
    return taskComplete, cm


def walkUp(cm, delta):
    idx = 1
    lowerFlag = 0
    motion.moveTo(0.2, 0, 0)
    while cm[0] < 420 and cm[0] > 0:
        pp = "ball_upfront"
        ext = ".png"
        im_num = pp + str(idx) + ext
        getImage(im_num, 0)
        img = cv2.imread(im_num)
        cm = CenterOfMassUp(img)
        if cm[0] == 0 and cm[1] == 0:
            # Scan the area with lower camera
            getImage('lower.png', 1)
            img = cv2.imread("lower.png")
            cm2 = CenterOfMassUp(img)
            lowerFlag = 1
            break
        else:
            alpha = (cm[1] - 320) * delta
            motion.moveTo(0.2, 0, alpha * 7 / 6)
            idx = idx + 1
            continue
    if lowerFlag == 1:
        if cm2[0] == 0 and cm2[1] == 0:
            lostFlag = 1
        else:
            lostFlag = 0
    else:
        getImage('lower.png', 1)
        img = cv2.imread('lower.png')
        cm2 = CenterOfMassUp(img)
        lostFlag = 0
    return lostFlag, cm2


def getReady(cm, delta):
    idx = 1
    pp = "ball_precise"
    ext = ".png"
    im_num = pp + str(idx - 1) + ext
    getImage(im_num, 1)
    img = cv2.imread(im_num)
    cm = CenterOfMassDown(img)
    alpha = (cm[1] - 320) * delta
    motion.moveTo(0, 0, alpha * 7 / 6)
    while cm[0] < 370:
        im_num = pp + str(idx) + ext
        getImage(im_num, 1)
        img = cv2.imread(im_num)
        cm = CenterOfMassDown(img)
        if cm == [0, 0]:
            return 0, cm
        if cm[0] < 405:
            alpha = (cm[1] - 320) * delta
            motion.moveTo(0.05, 0, alpha)
        else:
            break
        idx = idx + 1
    return 1


def isBallInHand():
    grabbed = False
    # if angles of right hand are >= 0.4 assume ball is grabbed.
    if (motion.getAngles("RHand", True)[0] >= 0.4):
        grabbed = True
        tts.say("I am holding the ball.")
    else:
        grabbed = False
        tts.say("I do not have the ball.")

    return grabbed


def pickUpBall():
    names = list()
    times = list()
    keys = list()

    names.append("HeadPitch")
    times.append([3, 5.4, 7.32, 9.52, 9.68, 11.08, 13.24, 13.32, 16.68, 18.24])
    keys.append([[0.0183661, [3, -1, 0], [3, 0.8, 0]], [-0.621311, [3, -0.8, 0], [3, 0.64, 0]],
                 [-0.589098, [3, -0.64, 0], [3, 0.733333, 0]], [-0.664264, [3, -0.733333, 0], [3, 0.0533333, 0]],
                 [-0.615176, [3, -0.0533333, -0.000876629], [3, 0.466667, 0.0076705]],
                 [-0.607505, [3, -0.466667, 0], [3, 0.72, 0]], [-0.653526, [3, -0.72, 0], [3, 0.0266667, 0]],
                 [-0.653526, [3, -0.0266667, 0], [3, 1.12, 0]],
                 [-0.0383921, [3, -1.12, -0.046256], [3, 0.52, 0.021476]], [-0.016916, [3, -0.52, 0], [3, 0, 0]]])

    names.append("HeadYaw")
    times.append([3, 5.4, 7.32, 9.52, 9.68, 11.08, 13.24, 13.32, 16.68, 18.24])
    keys.append([[0.00916195, [3, -1, 0], [3, 0.8, 0]], [0.0137641, [3, -0.8, -6.50186e-09], [3, 0.64, 5.20149e-09]],
                 [0.0137641, [3, -0.64, -5.20149e-09], [3, 0.733333, 5.96004e-09]],
                 [0.0214341, [3, -0.733333, 0], [3, 0.0533333, 0]], [0.0214341, [3, -0.0533333, 0], [3, 0.466667, 0]],
                 [0.0137641, [3, -0.466667, 0], [3, 0.72, 0]], [0.0137641, [3, -0.72, 0], [3, 0.0266667, 0]],
                 [0.0137641, [3, -0.0266667, 0], [3, 1.12, 0]], [0.0214341, [3, -1.12, 0], [3, 0.52, 0]],
                 [0.00916195, [3, -0.52, 0], [3, 0, 0]]])

    names.append("LAnklePitch")
    times.append([3, 5.4, 7.32, 9.52, 9.68, 11.08, 13.24, 13.32, 16.68, 18.24])
    keys.append([[-1.18944, [3, -1, 0], [3, 0.8, 0]], [-0.745566, [3, -0.8, -0.136457], [3, 0.64, 0.109166]],
                 [-0.452572, [3, -0.64, -0.00133875], [3, 0.733333, 0.00153398]],
                 [-0.451038, [3, -0.733333, -0.00153398], [3, 0.0533333, 0.000111562]],
                 [-0.4403, [3, -0.0533333, 0], [3, 0.466667, 0]],
                 [-0.563021, [3, -0.466667, 0.0446411], [3, 0.72, -0.0688749]],
                 [-0.780848, [3, -0.72, 0], [3, 0.0266667, 0]], [-0.780848, [3, -0.0266667, 0], [3, 1.12, 0]],
                 [-0.357464, [3, -1.12, -0.00991231], [3, 0.52, 0.00460214]], [-0.352862, [3, -0.52, 0], [3, 0, 0]]])

    names.append("LAnkleRoll")
    times.append([3, 5.4, 7.32, 9.52, 9.68, 11.08, 13.24, 13.32, 16.68, 18.24])
    keys.append([[0.023052, [3, -1, 0], [3, 0.8, 0]], [-0.0260359, [3, -0.8, 0], [3, 0.64, 0]],
                 [0.16418, [3, -0.64, 0], [3, 0.733333, 0]], [0.153442, [3, -0.733333, 0], [3, 0.0533333, 0]],
                 [0.153442, [3, -0.0533333, 0], [3, 0.466667, 0]], [0.158044, [3, -0.466667, 0], [3, 0.72, 0]],
                 [0.158044, [3, -0.72, 0], [3, 0.0266667, 0]], [0.158044, [3, -0.0266667, 0], [3, 1.12, 0]],
                 [0.0061779, [3, -1.12, 0.0099119], [3, 0.52, -0.00460196]], [0.00157595, [3, -0.52, 0], [3, 0, 0]]])

    names.append("LElbowRoll")
    times.append([3, 5.4, 7.32, 9.52, 9.68, 11.08, 13.24, 13.32, 16.68, 18.24])
    keys.append([[-0.987855, [3, -1, 0], [3, 0.8, 0]], [-0.300622, [3, -0.8, 0], [3, 0.64, 0]],
                 [-0.309826, [3, -0.64, 0], [3, 0.733333, 0]],
                 [-0.053648, [3, -0.733333, -0.0834167], [3, 0.0533333, 0.00606667]],
                 [-0.0413761, [3, -0.0533333, 0], [3, 0.466667, 0]],
                 [-0.338973, [3, -0.466667, 0.00298194], [3, 0.72, -0.00460071]],
                 [-0.343573, [3, -0.72, 0], [3, 0.0266667, 0]], [-0.343573, [3, -0.0266667, 0], [3, 1.12, 0]],
                 [-0.108872, [3, -1.12, 0], [3, 0.52, 0]], [-0.417486, [3, -0.52, 0], [3, 0, 0]]])

    names.append("LElbowYaw")
    times.append([3, 5.4, 7.32, 9.52, 9.68, 11.08, 13.24, 13.32, 16.68, 18.24])
    keys.append([[-1.37144, [3, -1, 0], [3, 0.8, 0]], [0.159494, [3, -0.8, 0], [3, 0.64, 0]],
                 [0.159494, [3, -0.64, 0], [3, 0.733333, 0]], [0.191708, [3, -0.733333, 0], [3, 0.0533333, 0]],
                 [0.184038, [3, -0.0533333, 0.000786665], [3, 0.466667, -0.00688332]],
                 [0.168698, [3, -0.466667, 0], [3, 0.72, 0]], [0.179436, [3, -0.72, 0], [3, 0.0266667, 0]],
                 [0.179436, [3, -0.0266667, 0], [3, 1.12, 0]], [-1.18276, [3, -1.12, 0.0364746], [3, 0.52, -0.0169346]],
                 [-1.19969, [3, -0.52, 0], [3, 0, 0]]])

    names.append("LHand")
    times.append([3, 5.4, 7.32, 9.52, 9.68, 11.08, 13.24, 13.32, 16.68, 18.24])
    keys.append([[0.246, [3, -1, 0], [3, 0.8, 0]], [0, [3, -0.8, 0], [3, 0.64, 0]],
                 [0.0164, [3, -0.64, -0.0164], [3, 0.733333, 0.0187916]],
                 [0.2272, [3, -0.733333, -0.0385007], [3, 0.0533333, 0.00280005]],
                 [0.23, [3, -0.0533333, 0], [3, 0.466667, 0]], [0.0196, [3, -0.466667, 0.00233333], [3, 0.72, -0.0036]],
                 [0.016, [3, -0.72, 0], [3, 0.0266667, 0]],
                 [0.0336, [3, -0.0266667, -0.00209923], [3, 1.12, 0.0881675]],
                 [0.2868, [3, -1.12, -0.0264289], [3, 0.52, 0.0122705]], [0.299071, [3, -0.52, 0], [3, 0, 0]]])

    names.append("LHipPitch")
    times.append([3, 5.4, 7.32, 9.52, 9.68, 11.08, 13.24, 13.32, 16.68, 18.24])
    keys.append([[-0.569072, [3, -1, 0], [3, 0.8, 0]], [-0.943368, [3, -0.8, 0.164479], [3, 0.64, -0.131583]],
                 [-1.45726, [3, -0.64, 0], [3, 0.733333, 0]], [-1.45726, [3, -0.733333, 0], [3, 0.0533333, 0]],
                 [-1.45726, [3, -0.0533333, 0], [3, 0.466667, 0]],
                 [-1.15353, [3, -0.466667, -0.120048], [3, 0.72, 0.185218]],
                 [-0.54146, [3, -0.72, 0], [3, 0.0266667, 0]], [-0.54146, [3, -0.0266667, 0], [3, 1.12, 0]],
                 [-0.455556, [3, -1.12, -0.0165195], [3, 0.52, 0.00766977]], [-0.447886, [3, -0.52, 0], [3, 0, 0]]])

    names.append("LHipRoll")
    times.append([3, 5.4, 7.32, 9.52, 9.68, 11.08, 13.24, 13.32, 16.68, 18.24])
    keys.append([[0.039926, [3, -1, 0], [3, 0.8, 0]], [-0.167164, [3, -0.8, 0.0553945], [3, 0.64, -0.0443156]],
                 [-0.259204, [3, -0.64, 0.00401626], [3, 0.733333, -0.00460196]],
                 [-0.263806, [3, -0.733333, 0], [3, 0.0533333, 0]], [-0.263806, [3, -0.0533333, 0], [3, 0.466667, 0]],
                 [-0.197844, [3, -0.466667, -0.0277499], [3, 0.72, 0.0428141]],
                 [-0.052114, [3, -0.72, 0], [3, 0.0266667, 0]], [-0.052114, [3, -0.0266667, 0], [3, 1.12, 0]],
                 [0.0061779, [3, -1.12, 0], [3, 0.52, 0]], [0.00310993, [3, -0.52, 0], [3, 0, 0]]])

    names.append("LHipYawPitch")
    times.append([3, 5.4, 7.32, 9.52, 9.68, 11.08, 13.24, 13.32, 16.68, 18.24])
    keys.append([[-0.612024, [3, -1, 0], [3, 0.8, 0]], [-0.993989, [3, -0.8, 0.0948807], [3, 0.64, -0.0759046]],
                 [-1.12438, [3, -0.64, 0.0120488], [3, 0.733333, -0.013806]],
                 [-1.13819, [3, -0.733333, 0], [3, 0.0533333, 0]], [-1.13819, [3, -0.0533333, 0], [3, 0.466667, 0]],
                 [-1.13665, [3, -0.466667, -0.00153345], [3, 0.72, 0.00236589]],
                 [-1.12591, [3, -0.72, 0], [3, 0.0266667, 0]], [-1.12591, [3, -0.0266667, 0], [3, 1.12, 0]],
                 [-0.00609398, [3, -1.12, 0], [3, 0.52, 0]], [-0.00762796, [3, -0.52, 0], [3, 0, 0]]])

    names.append("LKneePitch")
    times.append([3, 5.4, 7.32, 9.52, 9.68, 11.08, 13.24, 13.32, 16.68, 18.24])
    keys.append([[2.10461, [3, -1, 0], [3, 0.8, 0]], [2.03711, [3, -0.8, 0], [3, 0.64, 0]],
                 [2.11255, [3, -0.64, 0], [3, 0.733333, 0]], [2.10921, [3, -0.733333, 0], [3, 0.0533333, 0]],
                 [2.10921, [3, -0.0533333, 0], [3, 0.466667, 0]], [2.11255, [3, -0.466667, 0], [3, 0.72, 0]],
                 [2.00029, [3, -0.72, 0], [3, 0.0266667, 0]], [2.00029, [3, -0.0266667, 0], [3, 1.12, 0]],
                 [0.699462, [3, -1.12, 0], [3, 0.52, 0]], [0.70253, [3, -0.52, 0], [3, 0, 0]]])

    names.append("LShoulderPitch")
    times.append([3, 5.4, 7.32, 9.52, 9.68, 11.08, 13.24, 13.32, 16.68, 18.24])
    keys.append([[1.42504, [3, -1, 0], [3, 0.8, 0]], [0.544529, [3, -0.8, 0], [3, 0.64, 0]],
                 [0.54913, [3, -0.64, -0.00460121], [3, 0.733333, 0.00527222]],
                 [0.605888, [3, -0.733333, 0], [3, 0.0533333, 0]], [0.605888, [3, -0.0533333, 0], [3, 0.466667, 0]],
                 [0.470897, [3, -0.466667, 0.00298306], [3, 0.72, -0.00460244]],
                 [0.466294, [3, -0.72, 0], [3, 0.0266667, 0]], [0.466294, [3, -0.0266667, 0], [3, 1.12, 0]],
                 [1.55697, [3, -1.12, 0], [3, 0.52, 0]], [1.46334, [3, -0.52, 0], [3, 0, 0]]])

    names.append("LShoulderRoll")
    times.append([3, 5.4, 7.32, 9.52, 9.68, 11.08, 13.24, 13.32, 16.68, 18.24])
    keys.append([[0.288349, [3, -1, 0], [3, 0.8, 0]], [-0.15651, [3, -0.8, 0], [3, 0.64, 0]],
                 [-0.00771189, [3, -0.64, 0], [3, 0.733333, 0]], [-0.2869, [3, -0.733333, 0], [3, 0.0533333, 0]],
                 [-0.250084, [3, -0.0533333, -0.00886312], [3, 0.466667, 0.0775523]],
                 [-0.0276539, [3, -0.466667, 0], [3, 0.72, 0]], [-0.0506639, [3, -0.72, 0], [3, 0.0266667, 0]],
                 [-0.0506639, [3, -0.0266667, 0], [3, 1.12, 0]], [0.283748, [3, -1.12, 0], [3, 0.52, 0]],
                 [0.176053, [3, -0.52, 0], [3, 0, 0]]])

    names.append("LWristYaw")
    times.append([3, 5.4, 7.32, 9.52, 9.68, 11.08, 13.24, 13.32, 16.68, 18.24])
    keys.append([[-0.00771189, [3, -1, 0], [3, 0.8, 0]], [-1.62301, [3, -0.8, 0], [3, 0.64, 0]],
                 [-1.54018, [3, -0.64, 0], [3, 0.733333, 0]], [-1.65369, [3, -0.733333, 0], [3, 0.0533333, 0]],
                 [-1.63835, [3, -0.0533333, 0], [3, 0.466667, 0]], [-1.79176, [3, -0.466667, 0], [3, 0.72, 0]],
                 [-1.76107, [3, -0.72, 0], [3, 0.0266667, 0]], [-1.76107, [3, -0.0266667, 0], [3, 1.12, 0]],
                 [-0.22554, [3, -1.12, -0.422851], [3, 0.52, 0.196324]], [0.0964535, [3, -0.52, 0], [3, 0, 0]]])

    names.append("RAnklePitch")
    times.append([3, 5.4, 7.32, 9.52, 9.68, 11.08, 13.24, 13.32, 16.68, 18.24])
    keys.append([[-1.18421, [3, -1, 0], [3, 0.8, 0]], [-0.719404, [3, -0.8, -0.125561], [3, 0.64, 0.100449]],
                 [-0.506178, [3, -0.64, 0], [3, 0.733333, 0]], [-0.515382, [3, -0.733333, 0], [3, 0.0533333, 0]],
                 [-0.515382, [3, -0.0533333, 0], [3, 0.466667, 0]],
                 [-0.615092, [3, -0.466667, 0.0412226], [3, 0.72, -0.0636006]],
                 [-0.829852, [3, -0.72, 0], [3, 0.0266667, 0]], [-0.829852, [3, -0.0266667, 0], [3, 1.12, 0]],
                 [-0.355846, [3, -1.12, 0], [3, 0.52, 0]], [-0.360449, [3, -0.52, 0], [3, 0, 0]]])

    names.append("RAnkleRoll")
    times.append([3, 5.4, 7.32, 9.52, 9.68, 11.08, 13.24, 13.32, 16.68, 18.24])
    keys.append([[0.046062, [3, -1, 0], [3, 0.8, 0]], [0.0445281, [3, -0.8, 0.00153397], [3, 0.64, -0.00122717]],
                 [-0.118076, [3, -0.64, 0.0026775], [3, 0.733333, -0.00306797]],
                 [-0.121144, [3, -0.733333, 0], [3, 0.0533333, 0]], [-0.121144, [3, -0.0533333, 0], [3, 0.466667, 0]],
                 [-0.161028, [3, -0.466667, 0.00663584], [3, 0.72, -0.0102382]],
                 [-0.171766, [3, -0.72, 0], [3, 0.0266667, 0]], [-0.171766, [3, -0.0266667, 0], [3, 1.12, 0]],
                 [0.0123138, [3, -1.12, 0], [3, 0.52, 0]], [0.00771189, [3, -0.52, 0], [3, 0, 0]]])

    names.append("RElbowRoll")
    times.append([3, 5.4, 7.32, 9.52, 9.68, 11.08, 13.24, 13.32, 16.68, 18.24])
    keys.append([[0.0349066, [3, -1, 0], [3, 0.8, 0]], [0.297638, [3, -0.8, -0.083311], [3, 0.64, 0.0666488]],
                 [0.484786, [3, -0.64, 0], [3, 0.733333, 0]], [0.19486, [3, -0.733333, 0], [3, 0.0533333, 0]],
                 [0.19486, [3, -0.0533333, 0], [3, 0.466667, 0]],
                 [0.200996, [3, -0.466667, -0.00361954], [3, 0.72, 0.00558443]],
                 [0.222472, [3, -0.72, 0], [3, 0.0266667, 0]], [0.222472, [3, -0.0266667, 0], [3, 1.12, 0]],
                 [0.04913, [3, -1.12, 0.0107405], [3, 0.52, -0.00498668]], [0.0441433, [3, -0.52, 0], [3, 0, 0]]])

    names.append("RElbowYaw")
    times.append([3, 5.4, 7.32, 9.52, 9.68, 11.08, 13.24, 13.32, 16.68, 18.24])
    keys.append([[1.36522, [3, -1, 0], [3, 0.8, 0]], [0.469363, [3, -0.8, 0.0230109], [3, 0.64, -0.0184087]],
                 [0.450954, [3, -0.64, 0.0184087], [3, 0.733333, -0.0210933]],
                 [-0.0583339, [3, -0.733333, 0], [3, 0.0533333, 0]], [-0.0583339, [3, -0.0533333, 0], [3, 0.466667, 0]],
                 [-0.0874801, [3, -0.466667, 0.00522824], [3, 0.72, -0.00806643]],
                 [-0.0982179, [3, -0.72, 0], [3, 0.0266667, 0]], [-0.0982179, [3, -0.0266667, 0], [3, 1.12, 0]],
                 [1.24863, [3, -1.12, 0], [3, 0.52, 0]], [1.18682, [3, -0.52, 0], [3, 0, 0]]])

    names.append("RHand")
    times.append([3, 5.4, 7.32, 9.52, 9.68, 11.08, 13.24, 13.32, 16.68, 18.24])
    keys.append([[0.16, [3, -1, 0], [3, 0.8, 0]], [0.9668, [3, -0.8, 0], [3, 0.64, 0]],
                 [0.954, [3, -0.64, 0.0128], [3, 0.733333, -0.0146666]],
                 [0.5712, [3, -0.733333, 0.241503], [3, 0.0533333, -0.0175638]],
                 [0.1768, [3, -0.0533333, 0.00192], [3, 0.466667, -0.0168]],
                 [0.16, [3, -0.466667, 0.00613483], [3, 0.72, -0.00946517]], [0.13, [3, -0.72, 0], [3, 0.0266667, 0]],
                 [0.17, [3, -0.0266667, -0.000542636], [3, 1.12, 0.0227907]],
                 [0.2, [3, -1.12, -0.0113821], [3, 0.52, 0.00528455]], [0.21, [3, -0.52, 0], [3, 0, 0]]])

    names.append("RHipPitch")
    times.append([3, 5.4, 7.32, 9.52, 9.68, 11.08, 13.24, 13.32, 16.68, 18.24])
    keys.append([[-0.589097, [3, -1, 0], [3, 0.8, 0]], [-1.24105, [3, -0.8, 0.0479379], [3, 0.64, -0.0383503]],
                 [-1.2794, [3, -0.64, 0], [3, 0.733333, 0]], [-1.27786, [3, -0.733333, 0], [3, 0.0533333, 0]],
                 [-1.27786, [3, -0.0533333, 0], [3, 0.466667, 0]],
                 [-1.11373, [3, -0.466667, -0.0727932], [3, 0.72, 0.11231]],
                 [-0.722556, [3, -0.72, 0], [3, 0.0266667, 0]], [-0.722556, [3, -0.0266667, 0], [3, 1.12, 0]],
                 [-0.454106, [3, -1.12, -1.86421e-06], [3, 0.52, 8.65527e-07]], [-0.454105, [3, -0.52, 0], [3, 0, 0]]])

    names.append("RHipRoll")
    times.append([3, 5.4, 7.32, 9.52, 9.68, 11.08, 13.24, 13.32, 16.68, 18.24])
    keys.append([[-0.240796, [3, -1, 0], [3, 0.8, 0]], [-0.0429101, [3, -0.8, -0.0906196], [3, 0.64, 0.0724957]],
                 [0.24855, [3, -0.64, 0], [3, 0.733333, 0]], [0.24855, [3, -0.733333, 0], [3, 0.0533333, 0]],
                 [0.24855, [3, -0.0533333, 0], [3, 0.466667, 0]],
                 [0.14884, [3, -0.466667, 0.037402], [3, 0.72, -0.057706]],
                 [-0.0367741, [3, -0.72, 0], [3, 0.0266667, 0]], [-0.0367741, [3, -0.0266667, 0], [3, 1.12, 0]],
                 [-0.00916195, [3, -1.12, -0.00768249], [3, 0.52, 0.00356687]],
                 [-0.00302602, [3, -0.52, 0], [3, 0, 0]]])

    names.append("RHipYawPitch")
    times.append([3, 5.4, 7.32, 9.52, 9.68, 11.08, 13.24, 13.32, 16.68, 18.24])
    keys.append([[-0.612024, [3, -1, 0], [3, 0.8, 0]], [-0.993989, [3, -0.8, 0.0948807], [3, 0.64, -0.0759046]],
                 [-1.12438, [3, -0.64, 0.0120488], [3, 0.733333, -0.013806]],
                 [-1.13819, [3, -0.733333, 0], [3, 0.0533333, 0]], [-1.13819, [3, -0.0533333, 0], [3, 0.466667, 0]],
                 [-1.13665, [3, -0.466667, -0.00153345], [3, 0.72, 0.00236589]],
                 [-1.12591, [3, -0.72, 0], [3, 0.0266667, 0]], [-1.12591, [3, -0.0266667, 0], [3, 1.12, 0]],
                 [-0.00609398, [3, -1.12, 0], [3, 0.52, 0]], [-0.00762796, [3, -0.52, 0], [3, 0, 0]]])

    names.append("RKneePitch")
    times.append([3, 5.4, 7.32, 9.52, 9.68, 11.08, 13.24, 13.32, 16.68, 18.24])
    keys.append([[2.11255, [3, -1, 0], [3, 0.8, 0]], [2.10623, [3, -0.8, 0.00259118], [3, 0.64, -0.00207295]],
                 [2.09855, [3, -0.64, 0], [3, 0.733333, 0]], [2.11236, [3, -0.733333, 0], [3, 0.0533333, 0]],
                 [2.11236, [3, -0.0533333, 0], [3, 0.466667, 0]], [2.10316, [3, -0.466667, 0], [3, 0.72, 0]],
                 [2.10316, [3, -0.72, 0], [3, 0.0266667, 0]], [2.10316, [3, -0.0266667, 0], [3, 1.12, 0]],
                 [0.688808, [3, -1.12, 0], [3, 0.52, 0]], [0.690342, [3, -0.52, 0], [3, 0, 0]]])

    names.append("RShoulderPitch")
    times.append([3, 5.4, 7.32, 9.52, 9.68, 11.08, 13.24, 13.32, 16.68, 18.24])
    keys.append([[1.42206, [3, -1, 0], [3, 0.8, 0]], [0.710284, [3, -0.8, 0], [3, 0.64, 0]],
                 [0.721022, [3, -0.64, 0], [3, 0.733333, 0]], [0.590632, [3, -0.733333, 0], [3, 0.0533333, 0]],
                 [0.590632, [3, -0.0533333, 0], [3, 0.466667, 0]], [0.526205, [3, -0.466667, 0], [3, 0.72, 0]],
                 [0.544613, [3, -0.72, 0], [3, 0.0266667, 0]], [0.544613, [3, -0.0266667, 0], [3, 1.12, 0]],
                 [1.03089, [3, -1.12, -0.248748], [3, 0.52, 0.11549]], [1.63733, [3, -0.52, 0], [3, 0, 0]]])

    names.append("RShoulderRoll")
    times.append([3, 5.4, 7.32, 9.52, 9.68, 11.08, 13.24, 13.32, 16.68, 18.24])
    keys.append([[-0.276162, [3, -1, 0], [3, 0.8, 0]], [-0.128898, [3, -0.8, -0.0596555], [3, 0.64, 0.0477244]],
                 [0.0459781, [3, -0.64, -0.0629089], [3, 0.733333, 0.0720831]],
                 [0.276078, [3, -0.733333, 0], [3, 0.0533333, 0]], [0.276078, [3, -0.0533333, 0], [3, 0.466667, 0]],
                 [0.185572, [3, -0.466667, 0], [3, 0.72, 0]], [0.210117, [3, -0.72, 0], [3, 0.0266667, 0]],
                 [0.210117, [3, -0.0266667, 0], [3, 1.12, 0]],
                 [-0.250084, [3, -1.12, 0.010661], [3, 0.52, -0.00494975]], [-0.255034, [3, -0.52, 0], [3, 0, 0]]])

    names.append("RWristYaw")
    times.append([3, 5.4, 7.32, 9.52, 9.68, 11.08, 13.24, 13.32, 16.68, 18.24])
    keys.append([[-0.0138481, [3, -1, 0], [3, 0.8, 0]], [1.33454, [3, -0.8, -0.011505], [3, 0.64, 0.00920402]],
                 [1.34374, [3, -0.64, -0.00920402], [3, 0.733333, 0.0105463]],
                 [1.72571, [3, -0.733333, 0], [3, 0.0533333, 0]], [1.72571, [3, -0.0533333, 0], [3, 0.466667, 0]],
                 [1.73491, [3, -0.466667, -0.00542962], [3, 0.72, 0.00837713]],
                 [1.76713, [3, -0.72, 0], [3, 0.0266667, 0]], [1.76713, [3, -0.0266667, 0], [3, 1.12, 0]],
                 [0.133416, [3, -1.12, 0.0747885], [3, 0.52, -0.0347232]], [0.0986927, [3, -0.52, 0], [3, 0, 0]]])

    motion.angleInterpolationBezier(names, times, keys)

    pickUpBallValue = 1

    return pickUpBallValue


def main():
    # The robot wakes up: sets Motor on and, if needed, goes to initial position.
    motion.wakeUp()
    # Check if ball is already in Nao's hands.
    isBallGrabbed = isBallInHand()
    # Flag to check if task is completed.
    taskCompleteFlag = 0
    while taskCompleteFlag == 0:
        ballPosition, delta, cameraIndexValue = first_scan()
        if cameraIndexValue == 0:
            zero_head()
            lost, CoM = walkUp(ballPosition, delta)
            if lost == 0:
                # Switch cameras
                time.sleep(0.2)
                video.stopCamera(0)
                video.startCamera(1)
                video.setActiveCamera(1)
                zero_head()
                # Walk to the ball using lower camera
                taskCompleteFlag, CoM1 = walkDown(CoM, delta)
                taskCompleteFlag = getReady(CoM1, delta)
            else:
                tts.say('I cannot find the ball.')
                motion.moveTo(-0.2, 0, 0)
        elif cameraIndexValue == 1:
            # Switch cameras
            time.sleep(0.2)
            video.stopCamera(0)
            video.startCamera(1)
            video.setActiveCamera(1)
            zero_head()
            # Walk to the ball using lower camera
            taskCompleteFlag, CoM1 = walkDown(ballPosition, delta)
            taskCompleteFlag = getReady(CoM1, delta)

    # Check if ball is already in Nao's hands.
    if isBallGrabbed == True:
        motion.rest()
    else:
        pickUpBallValue = pickUpBall()
        if pickUpBallValue == 1:
            isBallInHand()

    motion.rest()


if __name__ == "__main__":
    main()