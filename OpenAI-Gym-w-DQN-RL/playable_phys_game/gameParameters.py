# Global Game Parameters
# gameParameters
# 210212 RJG Readme and minor mods
#

from vector2d import *
from pygame.locals import *  #

includeSound = False # 0 will deactivate any attempt to generate sound

# Number of ships shown in the game space
startNumShips = 3

bhGravityEventHorizonRadius = 20.   # Used for graphics and end of life
bhGravityMinRadius = 20.            # Used for gravity
shipGravityMinRadius = 5.           # Used for gravity

shipPositions = [] # Updated in movement loop  asteroids.py - playGame() loop

# Up to 5 spaceships
shipKeySteering = [
    (K_LEFT,K_RIGHT,K_UP)  #0
    ,(K_z,K_x,K_s)          #1
    ,(K_c,K_v,K_f)          #2
    ,(K_n,K_m,K_j)          #3
    ,(K_COMMA,K_PERIOD,K_l) #4
]

shipColor = [  # r,g,b
    ( 255, 20, 20 )
    ,( 20, 255, 20 )
    ,( 50, 50, 255)
    ,( 255, 255, 50)
    ,( 50, 255, 255)
]

# Screen parameters
screenWidth = 1024
screenHeight = 768
screenCenter = Vector2d( screenWidth/2. , screenHeight/2. )
