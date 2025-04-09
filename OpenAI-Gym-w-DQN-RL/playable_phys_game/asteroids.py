#!/usr/bin/env python3
#
# ***********************************************************************************
#    THIS IS THE FILE TO RUN TO START THE GAME
#    See readme.md
# ***********************************************************************************
#
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#  ===========================================================================================
#   Work on Gravity-based spaceship simulation
#    2020, 2021 Rick Gaitskell
#  ===========================================================================================
#   Portions and structure of this code are based on the PyGame Python-based Asteroids work of
#    Copyright (C) 2008  Nick Redshaw
#    Copyright (C) 2018  Francisco Sanchez Arroyo
#  ===========================================================================================
#
# 211217 Gaitskell
#  Demo working with 3 spacecraft under simple key control
#
# 210211 RJG
#   ALL THE LIBRARIES NEEDED SHOULD BE AVAILABLE THROUGH PIP
#   TESTED IN PYTHON 3.9
# 210212 RJG Readme and minor mods
#
# Notes:
# See README.md in project for key instructions
#
# Special commands
# ESC or Usual OS Quit command should end game
# P for pause
# R for toggle showing FPS
# O for frame advance whilst paused

import pygame
from pygame.locals import *

import sys
import os
import random
import math

# Part of package
from vectorsprites import *
from ship import *
from stage import *
from blackhole import *
from soundManager import *

import gameParameters

class Asteroids():

    def __init__(self):
        # Settings for splash screen

        self.stage = Stage('Spaceship Gravity Game', (gameParameters.screenWidth, gameParameters.screenHeight))
        self.paused = False
        self.showingFPS = True
        self.frameAdvance = False
        self.gameState = "attract_mode"

        self.blackholeList = []
        self.createBlackhole(1) # For splash screen

        self.secondsCount = 1
        self.score = 0
        self.ship = None
        self.shipsNum = 0       # Counter
        self.initialiseGame() # Added this to get game going immediately (without need for RETURN)



    def initialiseGame(self):
        # Settings for game

        self.gameState = 'playing'

        self.startNumShips = gameParameters.startNumShips
        # self.createNewOriginalShip()
        self.createShipsList()

        self.blackholeList = []
        self.numBlackhole = 1 # For game
        self.createBlackhole(self.numBlackhole)

        # self.score = 0
        # self.nextLife = 10000

        self.secondsCount = 1

    '''
    # This is no longer used
    def createNewOriginalShip(self):
        self.ship = Ship( self.stage , color=gameParameters.shipColor[0] )

        self.stage.addSprite( self.ship.thrustJet )
        self.stage.addSprite( self.ship )
    '''

    def createShipsList(self):
        self.shipsNum = 0
        self.shipsList = []

        for i in range(0, self.startNumShips):
            self.addShip()


    def addShip(self):
        self.shipsNum += 1
        print( self.shipsNum )
        if self.shipsNum <= len( gameParameters.shipColor ):
            self.ship = Ship( self.stage , shipIndex = self.shipsNum-1 , color=gameParameters.shipColor[self.shipsNum-1] ) # Creating new ships
        else:
            self.ship = Ship( self.stage , shipIndex = self.shipsNum-1 ) # If color info not available

        self.stage.addSprite( self.ship.thrustJet )
        self.stage.addSprite( self.ship )

        # ship.position.x = ( self.stage.width - (lifeNumber * ship.boundingRect.width) - 10 )
        # ship.position.y = ( 0 + ship.boundingRect.height )

        self.shipsList.append( self.ship )


    def createBlackhole(self , numBlackhole ):
        for _ in range( 0, numBlackhole ):
            # position = Vector2d(random.randrange(-10, 10), random.randrange(-10, 10))
            position = gameParameters.screenCenter # Assume only creating one in middle of game space

            newBlackhole = Blackhole(self.stage, position)
            self.stage.addSprite(newBlackhole)
            self.blackholeList.append(newBlackhole)


    # This is the main method that is running when game is in progress, or ready to play
    def playGame(self):

        clock = pygame.time.Clock()

        frameCount = 0.0
        timePassed = 0.0
        self.fps = 0.0

        # *******************  Main loop  ************************
        while True:

            # calculate fps
            timePassed += clock.tick(60) # Called once per frame, Execution will delay here if running faster than 60 fps
            frameCount += 1
            if frameCount % 10 == 0:  # every 10 frames
                # nearest integer
                self.fps = round((frameCount / (timePassed / 1000.0)))
                # reset counter
                timePassed = 0
                frameCount = 0

            self.secondsCount += 1

            self.input(pygame.event.get())

            # pause
            if self.paused and not self.frameAdvance:
                self.displayPaused()
                continue # returns the control to the beginning of the while loop

            # Collect Spaceship positions so that can calculate their mutual gravitation attraction
            gameParameters.shipPositions = [ self.shipsList[i].position for i in range(len(self.shipsList)) ]

            # Calculate movements of each sprite (spaceships)
            self.stage.screen.fill((10, 10, 10))
            self.stage.moveSprites()
            self.stage.drawSprites()
            self.displayScore()


            if self.showingFPS:
                self.displayFps()  # for debug


            # Process keys
            if self.gameState == 'playing':
                self.playing()    # ************************

            elif self.gameState == 'exploding':
                self.exploding()

            else:
                self.displayText()

            # Double buffer draw
            pygame.display.flip()



    def playing(self):
        if self.shipsNum == 0:
            self.gameState = 'attract_mode'
        else:
            self.processKeys()



    # move this kack somewhere else!
    def displayText(self):
        font1 = pygame.font.Font('Hyperspace.otf', 50)
        font2 = pygame.font.Font('Hyperspace.otf', 20)
        font3 = pygame.font.Font('Hyperspace.otf', 30)

        titleText = font1.render('Asteroids', True, (180, 180, 180))
        titleTextRect = titleText.get_rect(centerx=self.stage.width/2)
        titleTextRect.y = self.stage.height/2 - titleTextRect.height*2
        self.stage.screen.blit(titleText, titleTextRect)

        keysText = font2.render(
            '(C) 1979 Atari INC.', True, (255, 255, 255))
        keysTextRect = keysText.get_rect(centerx=self.stage.width/2)
        keysTextRect.y = self.stage.height - keysTextRect.height - 20
        self.stage.screen.blit(keysText, keysTextRect)

        instructionText = font3.render(
            'Press start to Play', True, (200, 200, 200))
        instructionTextRect = instructionText.get_rect(
            centerx=self.stage.width/2)
        instructionTextRect.y = self.stage.height/2 - instructionTextRect.height
        self.stage.screen.blit(instructionText, instructionTextRect)



    def displayScore(self):
        font1 = pygame.font.Font('Hyperspace.otf', 30)
        scoreStr = str("%02d" % self.score)
        scoreText = font1.render(scoreStr, True, (200, 200, 200))
        scoreTextRect = scoreText.get_rect(centerx=100, centery=45)
        self.stage.screen.blit(scoreText, scoreTextRect)



    def displayPaused(self):
        if self.paused:
            font1 = pygame.font.Font('Hyperspace.otf', 30)
            pausedText = font1.render("Paused", True, (255, 255, 255))
            textRect = pausedText.get_rect(
                centerx=self.stage.width/2, centery=self.stage.height/2)
            self.stage.screen.blit(pausedText, textRect)
            pygame.display.update()



    # Should move the ship controls into the ship class
    def input(self, events):
        self.frameAdvance = False

        # ************** THIS IS THE MAIN KEYBOARD EVENT HANDLER BUT NOT CONTROLS FOR SHIPS
        for event in events:
            if event.type == QUIT:
                sys.exit(0)

            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    sys.exit(0)

                if self.gameState == 'playing':
                    pass

                elif self.gameState == 'attract_mode':
                    # Start a new game
                    if event.key == K_RETURN:
                        self.initialiseGame()


                if event.key == K_p:
                    if self.paused:  # (is True)
                        self.paused = False
                    else:
                        self.paused = True

                if event.key == K_r:
                    if self.showingFPS:  # (is True)
                        self.showingFPS = False
                    else:
                        self.showingFPS = True

                # No fullscreen
                # if event.key == K_f:
                #    pygame.display.toggle_fullscreen()

            elif event.type == KEYUP: # RG - Don't know what this event is???
                if event.key == K_o:
                    self.frameAdvance = True



# Check Keyboard for Ship Controls
    def processKeys(self):
        key = pygame.key.get_pressed()

        for i in range(0 , min( len(gameParameters.shipKeySteering) , len(self.shipsList) ) ):
            if key[ gameParameters.shipKeySteering[i][0]] :
                self.shipsList[i].rotateLeft()
            elif key[ gameParameters.shipKeySteering[i][1]]:
                self.shipsList[i].rotateRight()
            if key[ gameParameters.shipKeySteering[i][2]]:
                self.shipsList[i].increaseThrust()
                self.shipsList[i].thrustJet.accelerating = True
            else:
                self.shipsList[i].thrustJet.accelerating = False

        '''
        # Ship 0 events
        i = 0
        if key[ gameParameters.shipKeySteering[i][0]]:
            self.ship.rotateLeft()
        elif key[ gameParameters.shipKeySteering[i][1]]:
            self.ship.rotateRight()
        if key[ gameParameters.shipKeySteering[i][2]]:
            self.ship.increaseThrust()
            self.ship.thrustJet.accelerating = True
        else:
            self.ship.thrustJet.accelerating = False
        '''



    def displayFps(self):
        font2 = pygame.font.Font('Hyperspace.otf', 15)
        fpsStr = str(self.fps)+(' FPS')
        scoreText = font2.render(fpsStr, True, (255, 255, 255))
        scoreTextRect = scoreText.get_rect(
            centerx=(self.stage.width/2), centery=15)
        self.stage.screen.blit(scoreText, scoreTextRect)


print('\n\n')
print('Basic instructions')
print('==================')
print('Spaceship controls #1 Arrows Left-Right-Up    #2 a-s-w    #3 j-k-i ')
print('ESC or Cmd-Q to Quit')

# Script to run the game
if not pygame.font:
    print('Warning, fonts disabled')
if not pygame.mixer:
    print('Warning, sound disabled')

initSoundManager()

game = Asteroids()  # create object game from class Asteroids
game.playGame()

####
