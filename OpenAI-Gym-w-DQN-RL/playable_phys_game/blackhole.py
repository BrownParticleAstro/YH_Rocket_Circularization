#
# 210212 RJG Readme and minor mods
#

import random
import math

from vectorsprites import *
from vector2d import *

from shooter import *
from soundManager import *
import gameParameters


# Blackhole

class Blackhole(VectorSprite):
    # indexes into the tuples below
    # Create the rock polygon to the given scale

    def __init__(self, stage , color=(0, 0, 255)): # Blue

        position = gameParameters.screenCenter
        heading = Vector2d(0,0)
        pointlist = self.createPointList()
        self.angle = 0. # Is this correct??

        VectorSprite.__init__(self, position, heading, pointlist)

    # BH pointlists
    def createPointList(self):
        # Create a circle polygon
        n = 8 # Number of separate points in circle
        radius = gameParameters.bhGravityEventHorizonRadius
        pointlist = [
            ( int(radius * math.cos(i / n * 2. * math.pi)), int(radius * math.sin(i / n * 2. * math.pi)) )
            for i in range(0, n+1) ] # Ensure it closes +1

        return pointlist

    def move(self):
        VectorSprite.move(self)
        # Spin the BH to make it look fun
        self.angle += 1

#    def destroyed(self):

'''
class Debris(Point):

    def __init__(self, position, stage):
        heading = Vector2d(random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5))
        Point.__init__(self, position, heading, stage)
        self.ttl = 50

    def move(self):
        Point.move(self)
        r, g, b = self.color
        r -= 5
        g -= 5
        b -= 5
        self.color = (r, g, b)
'''

# end
