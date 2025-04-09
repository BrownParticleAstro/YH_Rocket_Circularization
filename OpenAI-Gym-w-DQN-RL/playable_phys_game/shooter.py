#
# 210212 RJG Readme and minor mods
#

import random
from vectorsprites import *


class Shooter(VectorSprite):

    def __init__(self, position, heading, pointlist, stage , color=(255, 255, 255)):

        # Run init of the Superclass on this self
        VectorSprite.__init__(self, position, heading, pointlist , color=color )

        self.stage = stage

