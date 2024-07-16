#!/usr/bin/env python3

class MCTSNode(object):

    """
    :param
    parent: the parent node of current node
    children: the children nodes of current node
    visitTimes: number of visits to the current node, for UCB
    qualityValue: number of wins of current node, for UCB
    action: the action represented by the current node
    subActions: after perform the action of the current node, all actions that can be performed next
    stepListFromLastS: forward steplist
    team: the team performs this action (home_team or away_team)
    subTeam: the team performs the next actions
    isSetup: Is the formation set
    """
    def __init__(self):
        self.parent = None
        self.children = []
        self.visitTimes = 0
        self.qualityValue = 0.0
        self.action = None
        self.subActions = None
        self.stepListFromLastS = None
        self.team = None
        self.subTeam = None
        self.isSetup = False

    # get/set
    def getParent(self):
        return self.parent

    def setParent(self, parent):
        self.parent = parent

    def getAction(self):
        return self.action

    def setAction(self, action):
        self.action = action

    def getChildren(self):
        return self.children

    def setChildren(self, sub_node):
        self.children.append(sub_node)

    def getVisitTimes(self):
        return self.visitTimes

    def setVisitTimes(self, times):
        self.visitTimes = times

    def getQualityValue(self):
        return self.qualityValue

    def setQualityValue(self, value):
        self.qualityValue = value

    def getSubActions(self):
        return self.subActions

    def setSubActions(self, subactions):
        self.subActions = subactions

    def getStepListFromLastS(self):
        return self.stepListFromLastS

    def setStepListFromLastS(self, stepList):
        self.stepListFromLastS = stepList

    def getTeam(self):
        return self.team

    def setTeam(self, team):
        self.team = team

    def getSubTeam(self):
        return self.subTeam

    def setSubTeam(self, subteam):
        self.subTeam = subteam