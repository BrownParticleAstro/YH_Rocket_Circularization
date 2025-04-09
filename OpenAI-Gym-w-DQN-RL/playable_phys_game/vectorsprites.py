#

import pygame
import sys
import os
import math
import random
from math import *
from vector2d import *
from geometry import *


class VectorSprite:

    def __init__(self, position, heading, pointlist, angle=0, color=(255, 255, 255)):
        self.position = position
        self.heading = heading
        self.angle = angle
        self.vAngle = 0
        self.pointlist = pointlist  # raw pointlist
        self.color = color
        self.ttl = 25

        #self.color = color = (random.randrange(40,255),random.randrange(40,255),random.randrange(40,255))

    # rotate each x,y coord by the angle, then translate it to the x,y position
    def rotateAndTransform(self):
        newPointList = [self.rotatePoint(point) for point in self.pointlist]
        self.transformedPointlist = [
            self.translatePoint(point) for point in newPointList]

    # draw the sprite
    def draw(self):
        self.rotateAndTransform()
        return self.transformedPointlist

    # translate each point to the current x, y position
    def translatePoint(self, point):
        newPoint = []
        newPoint.append(point[0] + self.position.x)
        newPoint.append(point[1] + self.position.y)
        return newPoint

    # Move the sprite by the velocity
    def move(self):
        # Apply velocity
        self.position.x = self.position.x + self.heading.x
        self.position.y = self.position.y + self.heading.y
        self.angle = self.angle + self.vAngle

        # needed?
        # self.rotateAndTransform()

    # Rotate a point by the given angle
    def rotatePoint(self, point):
        newPoint = []
        cosVal = math.cos(radians(self.angle))
        sinVal = math.sin(radians(self.angle))
        newPoint.append(point[0] * cosVal + point[1] * sinVal)
        newPoint.append(point[1] * cosVal - point[0] * sinVal)

        # Keep points as integers
        newPoint = [int(point) for point in newPoint]
        return newPoint

    # Scale a point
    def scale(self, point, scale):
        newPoint = []
        newPoint.append(point[0] * scale)
        newPoint.append(point[1] * scale)
        # Keep points as integers
        newPoint = [int(point) for point in newPoint]
        return newPoint

    def collidesWith(self, target):
        if self.boundingRect.colliderect(target.boundingRect):
            return True
        else:
            return False

    # Check each line from pointlist1 for intersection with
    # the lines in pointlist2
    def checkPolygonCollision(self, target):
        for i in range(0, len(self.transformedPointlist)):
            for j in range(0, len(target.transformedPointlist)):
                p1 = self.transformedPointlist[i-1]
                p2 = self.transformedPointlist[i]
                p3 = target.transformedPointlist[j-1]
                p4 = target.transformedPointlist[j]
                p = calculateIntersectPoint(p1, p2, p3, p4)
                if (p != None):
                    return p

        return None
