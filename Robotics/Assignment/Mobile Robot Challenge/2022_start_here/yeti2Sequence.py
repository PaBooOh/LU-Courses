#!/usr/bin/env python
# coding: Latin-1

# Simple example of a motor sequence script

# Import library functions we need
import ZeroBorg
import time
import math
import sys

# Setup the ZeroBorg
ZB = ZeroBorg.ZeroBorg()
#ZB.i2cAddress = 0x44                  # Uncomment and change the value if you have changed the board address
ZB.Init()
if not ZB.foundChip:
    boards = ZeroBorg.ScanForZeroBorg()
    if len(boards) == 0:
        print 'No ZeroBorg found, check you are attached :)'
    else:
        print 'No ZeroBorg at address %02X, but we did find boards:' % (ZB.i2cAddress)
        for board in boards:
            print '    %02X (%d)' % (board, board)
        print 'If you need to change the I²C address change the setup line so it is correct, e.g.'
        print 'ZB.i2cAddress = 0x%02X' % (boards[0])
    sys.exit()
#ZB.SetEpoIgnore(True)                 # Uncomment to disable EPO latch, needed if you do not have a switch / jumper
ZB.SetCommsFailsafe(False)             # Disable the communications failsafe
ZB.ResetEpo()

# Movement settings (worked out from our YetiBorg v2 on a smooth surface)
timeForward1m = 2.4                     # Number of seconds needed to move about 1 meter
timeSpin360   = 2.2                     # Number of seconds needed to make a full left / right spin
testMode = True                        # True to run the motion tests, False to run the normal sequence

# Power settings
voltageIn = 8.4                         # Total battery voltage to the ZeroBorg (change to 9V if using a non-rechargeable battery)
voltageOut = 6.0                        # Maximum motor voltage

# Setup the power limits
if voltageOut > voltageIn:
    maxPower = 1.0
else:
    maxPower = voltageOut / float(voltageIn)

# Function to perform a general movement
def PerformMove(driveLeft, driveRight, numSeconds):
    # Set the motors running
    ZB.SetMotor1(-driveRight * maxPower) # Rear right
    ZB.SetMotor2(-driveRight * maxPower) # Front right
    ZB.SetMotor3(-driveLeft  * maxPower) # Front left
    ZB.SetMotor4(-driveLeft  * maxPower) # Rear left
    # Wait for the time
    time.sleep(numSeconds)
    # Turn the motors off
    ZB.MotorsOff()

# Function to spin an angle in degrees
def PerformSpin(angle):
    if angle < 0.0:
        # Left turn
        driveLeft  = -1.0
        driveRight = +1.0
        angle *= -1
    else:
        # Right turn
        driveLeft  = +1.0
        driveRight = -1.0
    # Calculate the required time delay
    numSeconds = (angle / 360.0) * timeSpin360
    # Perform the motion
    PerformMove(driveLeft, driveRight, numSeconds)

# Function to drive a distance in meters
def PerformDrive(meters):
    if meters < 0.0:
        # Reverse drive
        driveLeft  = -1.0
        driveRight = -1.0
        meters *= -1
    else:
        # Forward drive
        driveLeft  = +1.0
        driveRight = +1.0
    # Calculate the required time delay
    numSeconds = meters * timeForward1m
    # Perform the motion
    PerformMove(driveLeft, driveRight, numSeconds)

# Run test mode if required
if testMode:
    # Show settings
    print 'Current settings are:'
    print '    timeForward1m = %f' % (timeForward1m)
    print '    timeSpin360 = %f' % (timeSpin360)
    # Check distance
    raw_input('Check distance, Press ENTER to start')
    print 'Drive forward 30cm'
    PerformDrive(+0.3)
    raw_input('Press ENTER to continue')
    print 'Drive reverse 30cm'
    PerformDrive(-0.3)
    # Check spinning
    raw_input('Check spinning, Press ENTER to continue')
    print 'Spinning left'
    PerformSpin(-360)
    raw_input('Press ENTER to continue')
    print 'Spinning Right'
    PerformSpin(+360)
    print 'Update the settings as needed, then test again or disable test mode'
    sys.exit(0)

### Our sequence of motion goes here ###

# Draw a 40cm square
for i in range(4):
    PerformDrive(+0.4)
    PerformSpin(+90)

# Move to the middle of the square
PerformSpin(+45)
distanceToOtherCorner = math.sqrt(0.4**2 + 0.4**2) # Pythagorean theorem
PerformDrive(distanceToOtherCorner / 2.0)
PerformSpin(-45)

# Spin each way inside the square
PerformSpin(+360)
PerformSpin(-360)

# Return to the starting point
PerformDrive(-0.2)
PerformSpin(+90)
PerformDrive(-0.2)
PerformSpin(-90)
