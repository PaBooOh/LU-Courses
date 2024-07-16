"""
"""

from enum import Enum, unique
from typing import List, Optional

from src.env    import VrepEnvironment
from src.agents import Pioneer
from src.disp   import Display
import settings
import time
import matplotlib.pyplot as plt
import numpy as np
from math import cos, radians

'''
Priority:

1. Detect doors at a distance
2. Pass door
3. Exploration if no door found
'''


@unique
class Action(Enum):
    Stop = 0
    Forward = 1
    Backward = 2
    Fastforward = 3
    Fastbackward = 4
    Left = 5
    Right = 6
    slightLeft = 7
    slightRight = 8
    DriveRight = 9
    DriveLeft = 10
    ReserveRight = 11
    ReserveLeft = 12

    def performAction(self, agent):
        if self == self.Forward:
            agent.change_velocity([2, 2])
        elif self == self.Backward:
            agent.change_velocity([-2, -2])
        elif self == self.Fastforward:
            agent.change_velocity([5, 5])
        elif self == self.Fastbackward:
            agent.change_velocity([-5, -5])
        elif self == self.Left:
            agent.change_velocity([-.4, .4])
        elif self == self.Right:
            agent.change_velocity([.4, -.4])
        elif self == self.slightRight:
            agent.change_velocity([0.1, -0.1])
        elif self == self.slightLeft:
            agent.change_velocity([-0.1, 0.1])
        elif self == self.DriveRight:
            agent.change_velocity([5, 4.4])
        elif self == self.DriveLeft:
            agent.change_velocity([4.4, 5])
        elif self == self.ReserveLeft:
            agent.change_velocity([-1, -5])
        elif self == self.ReserveRight:
            agent.change_velocity([-5, -1])
        elif self == self.Stop:
            return
    
    def trackingOptimalDoor(agent, deg):
        left_wheel_speed = 3 + deg / 60.0
        right_wheel_speed = 3 - deg / 60.0
        agent.change_velocity([left_wheel_speed, right_wheel_speed])

class Door():
    Trajactory: List[float]
    startingDegree : Optional[int]

    def setTrajactory(self, lst):
        self.Trajactory = lst
    
    def setStartingDegree(self, deg):
        self.startingDegree = deg
    
    def getVariance(self):
        return np.var(self.Trajactory)
    
    def getMean(self):
        return np.mean(self.Trajactory)
    
    def getSum(self):
        return np.sum(self.Trajactory)
    
    def greaterRightTraj(self):
        traj_len = len(self.Trajactory)
        mid_idx = traj_len // 2
        left_mean = np.mean(self.Trajactory[:mid_idx])
        right_mean = np.mean(self.Trajactory[mid_idx:])
        if right_mean > left_mean:
            return True
        return False

# class Wall():


class Tracking():
    
    def findOutliersSegments(lidar_info):
        lidar_info[134] = (lidar_info[133] + lidar_info[135]) / 2 # calibrate 0 deg
        prev_dist = lidar_info[0]
        door = Door()
        door_trajactory = []
        doors = []

        for idx, dist in enumerate(lidar_info):
            if idx == 0:
                door_trajactory.append(dist)
                prev_dist = dist
                door.setStartingDegree(134)
                continue
            diff = abs(dist - prev_dist)
            if idx == len(lidar_info) - 1 and len(door_trajactory) > 0:
                door_trajactory.append(dist)
                door.setTrajactory(door_trajactory)
                doors.append(door)
                door_trajactory = []
                break
            if diff < 2:
                prev_dist = dist
                door_trajactory.append(dist)
                continue
            elif diff >= 2:
                door.setTrajactory(door_trajactory)
                doors.append(door)
                door = Door()
                door_trajactory = []
                door_trajactory.append(dist)
                door.setStartingDegree(134 - idx)
            prev_dist = dist
        
        return doors

def law_of_cosine(e1, e2, deg):
    rad = radians(deg)
    e3 = e1 ** 2 + e2 ** 2 - 2 * e1 * e2 * cos(rad)
    e3 = e3 ** 0.5
    return e3

def loop(agent):
    lidar_info = agent.read_lidars() # [0~134] : 134 ~ 0 degrees 
    outliers_segments = Tracking.findOutliersSegments(lidar_info)

    # Detect "possible doors" and select the optimal one
    optimal_doors = []
    optimal_door = None
    pass_doors = []
    for segment in outliers_segments:
        seg_trajactory = segment.Trajactory
        if len(seg_trajactory) >= 2 and np.mean(seg_trajactory) <= 1:
            pass_doors.append(segment)
            # print('Prepare passing door...')
            continue
        if len(seg_trajactory) <= 7 and np.mean(seg_trajactory) <= 6:
            continue
        if len(seg_trajactory) <= 4:
            continue
        if np.mean(seg_trajactory) <= 4.5:
            continue
        optimal_doors.append(segment)
    
    if len(pass_doors) > 0:
        for pass_door in pass_doors:
            if pass_door.startingDegree <= 135:
                if abs(agent.current_speed_API()[0]) and abs(agent.current_speed_API()[0]) < 0.05:
                    Action.performAction(Action.ReserveRight, agent)
                    print('Stuck at a door ... Try reserve...')
                    return
                Action.performAction(Action.DriveRight, agent)
                # Action.performAction(Action.Forward, agent)
                print('......Passing door......')
                return
    
    # If stuck
    if abs(agent.current_speed_API()[0]) and abs(agent.current_speed_API()[0]) < 0.05:
        print('Stuck!')
        Action.performAction(Action.Backward, agent)
        return

    # Exploration
    if len(optimal_doors) == 0:
        print('Noor Optimal door! Exploring...')
        Action.performAction(Action.Forward, agent)
        return

    auto_flag = 1
    if len(optimal_doors) >= 2:
        doors_ranked_by_mean = sorted(optimal_doors, key=lambda door: door.getMean())
        door_1 = doors_ranked_by_mean[-1].Trajactory # max
        door_2 = doors_ranked_by_mean[-2].Trajactory
        end_door_1_idx = 134 - (doors_ranked_by_mean[-1].startingDegree - len(door_1))
        end_door_2_idx = 134 - (doors_ranked_by_mean[-2].startingDegree - len(door_2))
        if np.mean(door_2) >= 8.5:
            diff_1 = np.mean(lidar_info[end_door_1_idx : end_door_1_idx + 10])
            diff_2 = np.mean(lidar_info[end_door_2_idx : end_door_2_idx + 10])
            # print('diff_1: ', diff_1)
            # print('diff_2: ', diff_2)
            if diff_2 <= diff_1:
                auto_flag = 0
                optimal_door = doors_ranked_by_mean[-1]
            else:
                auto_flag = 0
                optimal_door = doors_ranked_by_mean[-2]
        for segment in optimal_doors:
            len_trajactory = len(segment.Trajactory)
            start_door_degree = segment.startingDegree
            central_door_degree = (segment.startingDegree - len_trajactory // 2)
            end_door_degree = segment.startingDegree - len_trajactory
            if segment.greaterRightTraj() and auto_flag != 0:
                starting_degree = segment.startingDegree
                start_idx = 134 - starting_degree
                if np.mean(lidar_info[:start_idx]) <= 4.5:
                    auto_flag = 1
                    print('......Automatic tracking......')
                    Action.trackingOptimalDoor(agent, central_door_degree)
                    # Action.trackingOptimalDoor(agent, central_door_degree)
                    # Action.performAction(Action.Fastforward, agent)
                    return
        if auto_flag != 0:
            door_with_max = max(optimal_doors, key=lambda door: door.getMean())
            if len(door_with_max.Trajactory) >= 50:
                doors_ranked_by_var = sorted(optimal_doors, key=lambda door: door.getMean())
                optimal_door = doors_ranked_by_var[-1] # choose door trajactory with lowest variance
            else:
                optimal_door = door_with_max
    elif len(optimal_doors) == 1:
        optimal_door = optimal_doors[0]
    elif len(optimal_doors) == 0: 
        Action.performAction(Action.Fastforward, agent) # Exploration
        return
     
    
    len_trajactory = len(optimal_door.Trajactory)
    start_door_degree = optimal_door.startingDegree
    central_door_degree = (optimal_door.startingDegree - len_trajactory // 2)
    end_door_degree = optimal_door.startingDegree - len_trajactory

    # print('*** Traj: ', optimal_door.Trajactory)
    # print('*** Start: ', start_door_degree)
    # print('*** Central: ', central_door_degree)
    # print('*** End: ', end_door_degree)
    # print('*** Length: ', len_trajactory)
    # print('*** Mean: ', np.mean(optimal_door.Trajactory))
    # print('*** Var: ', np.var(optimal_door.Trajactory))
    # print()
    # for segment in outliers_segments:
    #     if len(segment.Trajactory) <= 4:
    #         continue
    #     print('Optimal: ', segment in optimal_doors)
    #     print(segment.Trajactory)
    #     print('Start_: ', segment.startingDegree)
    #     print('Len_: ', len(segment.Trajactory))
    #     print('Mean_: ', np.mean(segment.Trajactory))
    #     print('Var_: ', np.var(segment.Trajactory))
    #     print()
    # for door in doors:
    #     print(door.Trajactory)
    #     print('Start_: ', door.startingDegree)
    # print(lidar_info)
    # print('===============================')
    # print('===============================')
    # print()

    if start_door_degree < 0 and end_door_degree < 0:
        Action.performAction(Action.Left, agent)
        return
    if start_door_degree > 0 and end_door_degree > 0:
        Action.performAction(Action.Right, agent)
        return
    if start_door_degree + end_door_degree >= 8:
        Action.performAction(Action.slightRight, agent)
        return
    if start_door_degree + end_door_degree <= -8:
        Action.performAction(Action.slightLeft, agent)
        return
    if central_door_degree <= 6 and central_door_degree >= -6:
        Action.performAction(Action.Fastforward, agent)
        # Action.trackingOptimalDoor(agent, central_door_degree)
        return 

    Action.trackingOptimalDoor(agent, central_door_degree)
    # Action.performAction(Action.Fastforward, agent)
    return
    

if __name__ == "__main__":
    plt.ion()
    # Initialize and start the environment
    environment = VrepEnvironment(settings.SCENES + '/room_static.ttt')  # Open the file containing our scene (robot and its environment)
    environment.connect()        # Connect python to the simulator's remote API
    agent   = Pioneer(environment)
    display = Display(agent, False) 

    print('\nDemonstration of Simultaneous Localization and Mapping using CoppeliaSim robot simulation software. \nPress "CTRL+C" to exit.\n')
    start = time.time()
    step  = 0
    done  = False
    environment.start_simulation()
    time.sleep(1)
    # state = 0
    try:    
        while step < settings.simulation_steps and not done:
            display.update()                      # Update the SLAM display
            loop(agent)                           # Control loop
            step += 1
    except KeyboardInterrupt:
        print('\n\nInterrupted! Time: {}s'.format(time.time()-start))
        
    display.close()
    environment.stop_simulation()
    environment.disconnect()
