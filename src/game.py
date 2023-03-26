from itertools import cycle
import random
import sys
import pygame
from pygame.locals import *
from src.utils import pre_processing, noise
import cv2
import numpy as np
PLAYER = (
        'assets/sprites/redbird-upflap.png',
        'assets/sprites/redbird-midflap.png',
        'assets/sprites/redbird-downflap.png',
    )
BACKGROUND = 'assets/sprites/background-day.png'
PIPE = 'assets/sprites/pipe-green.png'

try:
    xrange
except NameError:
    xrange = range

FPS = 40
SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512
PIPE_GAP_SIZE = 100
BASEY = SCREEN_HEIGHT * 0.79
BIRD_HEIGHT = 24
PLAYER_PARAMS = {'MaxVelY': 10, 'MinVelY': -8, 'AccY': 1, 'RotVel': 3, 'RotThr': 20, 'FlapAcc': -9,
                 'FlapVel': -8, 'VelThr': 10}
BASE_WIDTH = 336
BACKGROUND_WIDTH = 288
BASE_SHIFT = BASE_WIDTH - BACKGROUND_WIDTH
PIPE_VELOCITY_X = -128 / FPS


class FlappyBird(object):

    def __init__(self):
        self.gaming = False
        self.score = 0
        self.action = 0
        self.IMAGES, self.HITMASKS = {}, {}
        pygame.init()
        self.FPSCLOCK = pygame.time.Clock()
        self.SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Flappy Bird')
        # numbers sprites for score display
        self.IMAGES['numbers'] = (
            pygame.image.load('assets/sprites/0.png').convert_alpha(),
            pygame.image.load('assets/sprites/1.png').convert_alpha(),
            pygame.image.load('assets/sprites/2.png').convert_alpha(),
            pygame.image.load('assets/sprites/3.png').convert_alpha(),
            pygame.image.load('assets/sprites/4.png').convert_alpha(),
            pygame.image.load('assets/sprites/5.png').convert_alpha(),
            pygame.image.load('assets/sprites/6.png').convert_alpha(),
            pygame.image.load('assets/sprites/7.png').convert_alpha(),
            pygame.image.load('assets/sprites/8.png').convert_alpha(),
            pygame.image.load('assets/sprites/9.png').convert_alpha()
        )

        # game over sprite
        self.IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
        # message sprite for welcome screen
        self.IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
        # base (ground) sprite
        self.IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

    def reset(self):
        self.IMAGES['background'] = pygame.image.load(BACKGROUND).convert()

        # select random player sprites
        self.IMAGES['player'] = (
            pygame.image.load(PLAYER[0]).convert_alpha(),
            pygame.image.load(PLAYER[1]).convert_alpha(),
            pygame.image.load(PLAYER[2]).convert_alpha(),
        )

        # select random pipe sprites
        self.IMAGES['pipe'] = (
            pygame.transform.rotate(
                pygame.image.load(PIPE).convert_alpha(), 180),
            pygame.image.load(PIPE).convert_alpha(),
        )

        # hismask for pipes
        self.HITMASKS['pipe'] = (
            self.getHitmask(self.IMAGES['pipe'][0]),
            self.getHitmask(self.IMAGES['pipe'][1]),
        )

        # hitmask for player
        self.HITMASKS['player'] = (
            self.getHitmask(self.IMAGES['player'][0]),
            self.getHitmask(self.IMAGES['player'][1]),
            self.getHitmask(self.IMAGES['player'][2]),
        )

        self.FPSCLOCK.tick(FPS)
        self.score = self.playerIndex = self.loopIter = 0
        self.player_x, self.player_y = int(SCREEN_WIDTH * 0.2), int((SCREEN_HEIGHT - BIRD_HEIGHT) / 2)
        self.playerIndexGen = cycle([0, 1, 2, 1])
        pygame.display.update()
        self.basex = -(4 % BASE_SHIFT)

        # get 2 new pipes to add to self.upperPipes self.lowerPipes list
        self.newPipe1 = self.getRandomPipe()
        self.newPipe2 = self.getRandomPipe()

        # list of upper pipes
        self.upperPipes = [
            {'x': SCREEN_WIDTH + 200, 'y': self.newPipe1[0]['y']},
            {'x': SCREEN_WIDTH + 200 + (SCREEN_WIDTH / 2), 'y': self.newPipe2[0]['y']},
        ]

        # list of lowerpipe
        self.lowerPipes = [
            {'x': SCREEN_WIDTH + 200, 'y': self.newPipe1[1]['y']},
            {'x': SCREEN_WIDTH + 200 + (SCREEN_WIDTH / 2), 'y': self.newPipe2[1]['y']},
        ]

        # player velocity, max velocity, downward accleration, accleration on flap
        self.playerVelY = -9  # player's velocity along Y, default same as self.playerFlapped
        self.playerRot = 45  # player's rotation
        self.playerFlapped = False  # True when player flaps

    def next_frame(self, action):
        reward = 0.1
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and (event.key == pygame.K_SPACE or event.key == pygame.K_UP):
                if self.player_y > -2 * self.IMAGES['player'][0].get_height():
                    self.playerVelY = PLAYER_PARAMS['FlapAcc']
                    self.playerFlapped = True
        if action:
            if self.player_y > -2 * self.IMAGES['player'][0].get_height():
                self.playerVelY = PLAYER_PARAMS['FlapAcc']
                self.playerFlapped = True
        playerMidPos = self.player_x + self.IMAGES['player'][0].get_width() / 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + self.IMAGES['pipe'][0].get_width() / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                self.score += 1
                reward = 1

                print("\033[1;32m Score! \033[0m")

        # self.playerIndex self.basex change
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(self.playerIndexGen)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % BASE_SHIFT)

        # rotate the player
        if self.playerRot > -90:
            self.playerRot -= PLAYER_PARAMS['RotVel']

        # player's movement
        if self.playerVelY < PLAYER_PARAMS['MaxVelY'] and not self.playerFlapped:
            self.playerVelY += PLAYER_PARAMS['AccY']
        if self.playerFlapped:
            self.playerFlapped = False

            # more rotation to cover the threshold (calculated in visible rotation)
            self.playerRot = 45

        playerHeight = self.IMAGES['player'][self.playerIndex].get_height()
        self.player_y += min(self.playerVelY, BASEY - self.player_y - playerHeight)

        # move pipes to left
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += PIPE_VELOCITY_X
            lPipe['x'] += PIPE_VELOCITY_X

        # add new pipe when first pipe is about to touch left of screen
        if 0 < self.upperPipes[0]['x'] < 5 and 0 < len(self.upperPipes) < 3:
            self.newPipe = self.getRandomPipe()
            self.upperPipes.append(self.newPipe[0])
            self.lowerPipes.append(self.newPipe[1])

        # remove first pipe if its out of the screen
        if self.upperPipes[0]['x'] < -self.IMAGES['pipe'][0].get_width() and len(self.upperPipes) > 0:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)
        backgroundSurface = pygame.transform.scale(self.IMAGES['background'], (SCREEN_WIDTH, SCREEN_HEIGHT))
        blackSurface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

        # draw sprites
        self.SCREEN.blit(backgroundSurface, (0, 0))

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            self.SCREEN.blit(self.IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            self.SCREEN.blit(self.IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))
            blackSurface.blit(self.IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            blackSurface.blit(self.IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        self.SCREEN.blit(self.IMAGES['base'], (self.basex, BASEY))
        # print score so player overlaps the score
        self.showScore()

        # Player rotation has a threshold
        visibleRot = PLAYER_PARAMS['RotThr']
        if self.playerRot <= PLAYER_PARAMS['RotThr']:
            visibleRot = self.playerRot

        if self.player_y <= 0:
            self.player_y = 0
            reward = -0.1
        playerSurface = pygame.transform.rotate(self.IMAGES['player'][self.playerIndex], visibleRot)
        self.SCREEN.blit(playerSurface, (self.player_x, self.player_y))
        blackSurface.blit(playerSurface, (self.player_x, self.player_y))

        image = pygame.surfarray.array3d(blackSurface)
        image = np.transpose(image, (1, 0, 2))
        image = image[:, :, [2, 1, 0]]
        image = pre_processing(image[:int(BASEY), :SCREEN_WIDTH], 84, 84)
        pygame.display.update()
        self.FPSCLOCK.tick(FPS)
        is_crashed = self.checkCrash({'x': self.player_x, 'y': self.player_y,
                                      'index': self.playerIndex},
                                     self.upperPipes, self.lowerPipes)
        if is_crashed:
            reward -= 5
        return image, reward, is_crashed

    def getRandomPipe(self):
        """returns a randomly generated pipe"""
        # y of gap between upper and lower pipe
        gapY = random.randrange(0, int(BASEY * 0.6 - PIPE_GAP_SIZE))
        gapY += int(BASEY * 0.2)
        pipeHeight = self.IMAGES['pipe'][0].get_height()
        pipeX = SCREEN_WIDTH + 10

        return [
            {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
            {'x': pipeX, 'y': gapY + PIPE_GAP_SIZE},  # lower pipe
        ]

    def showScore(self):
        """displays score in center of screen"""
        scoreDigits = [int(x) for x in list(str(self.score))]
        totalWidth = 0

        for digit in scoreDigits:
            totalWidth += self.IMAGES['numbers'][digit].get_width()

        Xoffset = (SCREEN_WIDTH - totalWidth) / 2

        for digit in scoreDigits:
            self.SCREEN.blit(self.IMAGES['numbers'][digit], (Xoffset, SCREEN_HEIGHT * 0.1))
            Xoffset += self.IMAGES['numbers'][digit].get_width()

    def checkCrash(self, player, upperPipes, lowerPipes):
        pi = player['index']
        player['w'] = self.IMAGES['player'][0].get_width()
        player['h'] = self.IMAGES['player'][0].get_height()

        # if player crashes into ground
        if player['y'] + player['h'] >= BASEY - 1:
            return True
        else:

            playerRect = pygame.Rect(player['x'], player['y'],
                                     player['w'], player['h'])
            pipeW = self.IMAGES['pipe'][0].get_width()
            pipeH = self.IMAGES['pipe'][0].get_height()

            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                # upper and lower pipe rects
                uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
                lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

                # player and upper/lower pipe hitmasks
                pHitMask = self.HITMASKS['player'][pi]
                uHitmask = self.HITMASKS['pipe'][0]
                lHitmask = self.HITMASKS['pipe'][1]

                # if bird collided with upipe or lpipe
                uCollide = self.pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
                lCollide = self.pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

                if uCollide or lCollide:
                    return True

        return False

    def pixelCollision(self, rect1, rect2, hitmask1, hitmask2):
        """Checks if two objects collide and not just their rects"""
        rect = rect1.clip(rect2)

        if rect.width == 0 or rect.height == 0:
            return False

        x1, y1 = rect.x - rect1.x, rect.y - rect1.y
        x2, y2 = rect.x - rect2.x, rect.y - rect2.y

        for x in xrange(rect.width):
            for y in xrange(rect.height):
                if hitmask1[x1 + x][y1 + y] and hitmask2[x2 + x][y2 + y]:
                    return True
        return False

    def getHitmask(self, image):
        """returns a hitmask using an image's alpha."""
        mask = []
        for x in xrange(image.get_width()):
            mask.append([])
            for y in xrange(image.get_height()):
                mask[x].append(bool(image.get_at((x, y))[3]))
        return mask
