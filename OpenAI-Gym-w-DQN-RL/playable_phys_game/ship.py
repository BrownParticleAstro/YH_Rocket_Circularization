#
# 210212 RJG Readme and minor mods
#

import random
from vectorsprites import *
from math import *
from shooter import *
from soundManager import *

import gameParameters

class Ship(Shooter): # Class is based on Shooter

    # Class attributes
    acceleration = 0.08   # 0.2 is too strong
    decelaration = 0.    # -0.005
    maxVelocity = 4
    turnAngle = 3        # 6

    massShip = 0.3 # Standard mass for ship relative to BH used for gravity calculation

    gravity_strength = 500.0 # Acceleration due to gravity at 1 pixels radius, scales 1/radius**2
    gravity_position_x = gameParameters.screenWidth / 2. # Middle, Based on size of screen
    gravity_position_y = gameParameters.screenHeight / 2.

    # bulletVelocity = 13.0
    # maxBullets = 4
    # bulletTtl = 35

    def __init__(self, stage , shipIndex , color = (255, 255, 255) ):
        # shipIndex is simple 0,1,2,

        self.shipIndex = shipIndex # Used when calculating gravity

        # May have to allow a position to be passed - use random for time being
        position = Vector2d( random.randrange(0, stage.width) , random.randrange(0, stage.height) )
        self.position = position
        #position = Vector2d(stage.width/4., stage.height/4.) # Start away from the BH

        heading = Vector2d(0, 0)

        self.thrustJet = ThrustJet(stage, self)
        self.thrustJet.color = (255, 255, 255)
        self.thrustJet.position = self.position

        self.visible = True
        self.color = color

        pointlist = [(0, -10), (6, 10), (3, 7), (-3, 7), (-6, 10)]

        # Run init of the Superclass on this self
        Shooter.__init__(self, position, heading, pointlist, stage , color=color )
        # Below is the Python3 equivalent
        # super().__init__( position, heading, pointlist, stage)

    def draw(self):
        if self.visible:
            VectorSprite.draw(self)
        return self.transformedPointlist


    def rotateLeft(self):
        self.angle += self.turnAngle
        self.thrustJet.angle += self.turnAngle
    def rotateRight(self):
        self.angle -= self.turnAngle
        self.thrustJet.angle -= self.turnAngle

    def increaseThrust(self):
        playSoundContinuous("thrust")
        if math.hypot(self.heading.x, self.heading.y) > self.maxVelocity:
            return

        dx = self.acceleration * math.sin(radians(self.angle)) * -1
        dy = self.acceleration * math.cos(radians(self.angle)) * -1
        self.changeVelocity(dx, dy)


    def applyBlackholeGravity(self):
        # Calculation of radius**2, don't allow it to become too small to limit strength of interaction
        radius2 = (self.position.x-self.gravity_position_x)**2 + (self.position.y-self.gravity_position_y)**2

        # Radius doesn't go below a threshold. So force close by is ~r rather than 1/r**2
        # Should this just turn force off completely?
        radius2 = max( radius2 , gameParameters.bhGravityMinRadius**2 )

        dx = - self.gravity_strength * (self.position.x-self.gravity_position_x) / radius2**1.5
        dy = - self.gravity_strength * (self.position.y-self.gravity_position_y) / radius2**1.5
        self.changeVelocity(dx, dy)


    def applyOtherShipsGravity(self):
        # gameParameters.shipPositions
        # Loop over every ship calculating intreaction strength
        for pvec in gameParameters.shipPositions:
            radius = ((self.position.x-pvec.x)**2+(self.position.y-pvec.y)**2)**0.5
            if radius < gameParameters.shipGravityMinRadius :
                return # Too close for gravitational interaction
                # Turns of force when too close
                # This ensures ship don't self interact
            dx = - self.massShip * self.gravity_strength * (self.position.x -pvec.x ) / radius ** 3.
            dy = - self.massShip * self.gravity_strength * (self.position.y -pvec.y ) / radius ** 3.
            self.changeVelocity(dx, dy)


    def decreaseThrust(self):
        stopSound("thrust")
        if (self.heading.x == 0 and self.heading.y == 0):
            return

        dx = self.heading.x * self.decelaration
        dy = self.heading.y * self.decelaration
        self.changeVelocity(dx, dy)


    def changeVelocity(self, dx, dy):
        self.heading.x += dx
        self.heading.y += dy
        self.thrustJet.heading.x += dx
        self.thrustJet.heading.y += dy


    def move(self):
        VectorSprite.move(self)
        self.applyBlackholeGravity() # ASSUME SINGLE TICK OF TIME - not trying to scale for different intervals of time
        self.applyOtherShipsGravity() # ASSUME SINGLE TICK OF TIME - not trying to scale for different intervals of time
        self.decreaseThrust() # Currently disabled


# Exhaust jet when ship is accelerating
class ThrustJet(VectorSprite):
    pointlist = [(-3, 7), (0, 13), (3, 7)]

    def __init__(self, stage, ship):
        self.ship = ship
        position = Vector2d(stage.width/4., stage.height/4.)
        # position = Vector2d(self.ship.position.x, self.ship.position.y)
        # position = ship.position
        # self.position = position # Do we need position here?

        heading = Vector2d(0, 0)
        self.accelerating = False
        VectorSprite.__init__(self, position, heading, self.pointlist , color = (0,0,0) )  # Black

    def draw(self):
        if self.accelerating:
            self.color = (255, 255, 255)
        else:
            self.color = (0, 0, 0)

        VectorSprite.draw(self)
        return self.transformedPointlist
