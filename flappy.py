from itertools import cycle
import random
import sys

import pygame
from pygame.locals import *

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

# DIFFICULTY: Easy Mode
FPS = 30
SCREENWIDTH  = 288.0
SCREENHEIGHT = 512.0
# amount by which base can maximum shift to left
PIPEGAPSIZE  = 130 # gap between upper and lower part of pipe (Easy: 130, Difficult: 120)
BASEY        = SCREENHEIGHT * 0.79
# image, sound and hitmask  dicts
IMAGES, SOUNDS, HITMASKS = {}, {}, {}

load_saved_pool = 1  # Desabilitar carregamento - começar do zero
save_current_pool = 1
current_pool = []
fitness = []
total_models = 10  # As per specification

next_pipe_x = -1
next_pipe_hole_y = -1
generation = 1  # Reset for training mode

def save_pool():
    for xi in range(total_models):
        current_pool[xi].save_weights("Current_Model_Pool/model_new" + str(xi) + ".weights.h5")
    print("Saved current pool!")

def model_crossover(model_idx1, model_idx2):
    global current_pool
    weights1 = current_pool[model_idx1].get_weights()
    weights2 = current_pool[model_idx2].get_weights()
    
    # Create deep copies to avoid modifying original weights
    weightsnew1 = [w.copy() for w in weights1]
    weightsnew2 = [w.copy() for w in weights2]
    
    # Improved crossover: mix weights from both parents more thoroughly
    for i in range(len(weightsnew1)):
        if len(weightsnew1[i].shape) == 2:  # Weight matrix
            # For each weight, randomly choose from parent1 or parent2
            mask = np.random.random(weightsnew1[i].shape) > 0.5
            weightsnew1[i] = np.where(mask, weights1[i], weights2[i])
            weightsnew2[i] = np.where(mask, weights2[i], weights1[i])
        else:  # Bias vector
            mask = np.random.random(weightsnew1[i].shape) > 0.5
            weightsnew1[i] = np.where(mask, weights1[i], weights2[i])
            weightsnew2[i] = np.where(mask, weights2[i], weights1[i])
    
    return [weightsnew1, weightsnew2]

def model_mutate(weights):
    mutation_rate = 0.2  # As per specification
    mutation_strength = 0.5  # Moderate changes for evolution
    
    for xi in range(len(weights)):
        # Handle different weight shapes (matrices and vectors)
        if len(weights[xi].shape) == 2:  # Weight matrix
            for yi in range(weights[xi].shape[0]):
                for zi in range(weights[xi].shape[1]):
                    if random.uniform(0, 1) < mutation_rate:
                        change = random.uniform(-mutation_strength, mutation_strength)
                        weights[xi][yi][zi] += change
        else:  # Bias vector
            for yi in range(weights[xi].shape[0]):
                if random.uniform(0, 1) < mutation_rate:
                    change = random.uniform(-mutation_strength, mutation_strength)
                    weights[xi][yi] += change
    return weights

def predict_action(player_y, player_x, pipes, model_num):
    global current_pool
    
    # Find the closest gap ahead of the player
    closest_pipe_x = float('inf')
    closest_gap_y = 0
    
    # Look for the next pipe that the player hasn't passed yet
    for i, (upper_pipe, lower_pipe) in enumerate(zip(pipes['upper'], pipes['lower'])):
        pipe_x = upper_pipe['x'] + IMAGES['pipe'][0].get_width()  # Right edge of pipe
        if pipe_x > player_x:  # Pipe is ahead of player
            if pipe_x < closest_pipe_x:
                closest_pipe_x = pipe_x
                # Calculate gap center
                gap_top = upper_pipe['y'] + IMAGES['pipe'][0].get_height()
                gap_bottom = lower_pipe['y']
                closest_gap_y = (gap_top + gap_bottom) / 2
            break
    
    if closest_pipe_x == float('inf'):
        # No pipes ahead, use default action
        return 2
    
    # Calculate inputs as per specification:
    # 1. Horizontal distance to closest gap
    horizontal_distance = closest_pipe_x - player_x
    # 2. Height difference to closest gap (positive = bird above gap, negative = bird below gap)
    height_difference = player_y - closest_gap_y
    
    # Normalize inputs to [-0.5, 0.5] range
    horizontal_distance = min(450, horizontal_distance) / 450 - 0.5
    height_difference = height_difference / SCREENHEIGHT
    
    neural_input = np.asarray([horizontal_distance, height_difference])
    neural_input = np.atleast_2d(neural_input)
    output_prob = current_pool[model_num].predict(neural_input, verbose=0)[0]
    
    if output_prob[0] > 0.5:
        # Perform the jump action
        return 1
    return 2

# Initialize all models
for i in range(total_models):
    model = Sequential()
    model.add(Dense(units=5, input_dim=2))  # 2 inputs, 5 hidden neurons (can be adjusted between 1-20)
    model.add(Activation("sigmoid"))
    model.add(Dense(units=1))
    model.add(Activation("sigmoid"))

    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss="mse", optimizer=sgd, metrics=["accuracy"])
    current_pool.append(model)
    fitness.append(0)  # Initialize fitness to 0

if load_saved_pool:
    for i in range(total_models):
        try:
            current_pool[i].load_weights("Current_Model_Pool/model_new"+str(i)+".weights.h5")
            print(f"Loaded weights for model {i}")
        except Exception as e:
            print(f"Could not load weights for model {i}: {e}")
            print("Starting with random weights for this model")

# Uncomment the line below if you want to see the model weights
# for i in range(total_models):
#     print(current_pool[i].get_weights())

# list of all possible players (tuple of 3 positions of flap)
PLAYERS_LIST = (
    # red bird
    (
        'assets/sprites/redbird-upflap.png',
        'assets/sprites/redbird-midflap.png',
        'assets/sprites/redbird-downflap.png',
    ),
    # blue bird
    (
        # amount by which base can maximum shift to left
        'assets/sprites/bluebird-upflap.png',
        'assets/sprites/bluebird-midflap.png',
        'assets/sprites/bluebird-downflap.png',
    ),
    # yellow bird
    (
        'assets/sprites/yellowbird-upflap.png',
        'assets/sprites/yellowbird-midflap.png',
        'assets/sprites/yellowbird-downflap.png',
    ),
)

# list of backgrounds
BACKGROUNDS_LIST = (
    'assets/sprites/background-day.png',
    'assets/sprites/background-night.png',
)

# list of pipes
PIPES_LIST = (
    'assets/sprites/pipe-green.png',
    'assets/sprites/pipe-red.png',
)


def main():
    global SCREEN, FPSCLOCK
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((int(SCREENWIDTH), int(SCREENHEIGHT)))
    pygame.display.set_caption('Flappy Bird')

    # numbers sprites for score display
    IMAGES['numbers'] = (
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
    IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
    # message sprite for welcome screen
    IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
    # base (ground) sprite
    IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

    # sounds
    if 'win' in sys.platform:
        soundExt = '.wav'
    else:
        soundExt = '.ogg'

    SOUNDS['die']    = pygame.mixer.Sound('assets/audio/die' + soundExt)
    SOUNDS['hit']    = pygame.mixer.Sound('assets/audio/hit' + soundExt)
    SOUNDS['point']  = pygame.mixer.Sound('assets/audio/point' + soundExt)
    SOUNDS['swoosh'] = pygame.mixer.Sound('assets/audio/swoosh' + soundExt)
    SOUNDS['wing']   = pygame.mixer.Sound('assets/audio/wing' + soundExt)

    while True:
        # select random background sprites
        randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
        IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()

        # select random player sprites
        randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
        IMAGES['player'] = (
            pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
        )

        # select random pipe sprites
        pipeindex = random.randint(0, len(PIPES_LIST) - 1)
        IMAGES['pipe'] = (
            pygame.transform.rotate(
                pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), 180),
            pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
        )

        # hismask for pipes
        HITMASKS['pipe'] = (
            getHitmask(IMAGES['pipe'][0]),
            getHitmask(IMAGES['pipe'][1]),
        )

        # hitmask for player
        HITMASKS['player'] = (
            getHitmask(IMAGES['player'][0]),
            getHitmask(IMAGES['player'][1]),
            getHitmask(IMAGES['player'][2]),
        )

        movementInfo = showWelcomeAnimation()
        global fitness
        for idx in range(total_models):
            fitness[idx] = 0
        crashInfo = mainGame(movementInfo)
        showGameOverScreen(crashInfo)


def showWelcomeAnimation():
    return {
                'playery': int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2),
                'basex': 0,
                'playerIndexGen': cycle([0, 1, 2, 1]),
            }


def mainGame(movementInfo):
    global fitness
    score = playerIndex = loopIter = 0
    playerIndexGen = movementInfo['playerIndexGen']
    playersXList = []
    playersYList = []
    individual_scores = [0] * total_models  # Track individual scores for each bird
    frame_count = 0  # Track frames for distance calculation
    for idx in range(total_models):
        playerx, playery = int(SCREENWIDTH * 0.2), movementInfo['playery']
        playersXList.append(playerx)
        playersYList.append(playery)
    basex = movementInfo['basex']
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # get 2 new pipes to add to upperPipes lowerPipes list
    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()

    # list of upper pipes
    upperPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
    ]

    # list of lowerpipe
    lowerPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
    ]

    global next_pipe_x
    global next_pipe_hole_y

    next_pipe_x = lowerPipes[0]['x']
    next_pipe_hole_y = (lowerPipes[0]['y'] + (upperPipes[0]['y'] + IMAGES['pipe'][0].get_height()))/2

    pipeVelX = -4  # Easy mode: velocity 200 (adjusted for game scale)

    # player velocity, max velocity, downward accleration, accleration on flap
    playersVelY    =  []   # player's velocity along Y, default same as playerFlapped
    playerMaxVelY =  10   # max vel along Y, max descend speed
    playerMinVelY =  -8   # min vel along Y, max ascend speed
    playersAccY    =  []   # players downward accleration
    playerFlapAcc =  -9   # players speed on flapping
    playersFlapped = [] # True when player flaps
    playersState = []

    for idx in range(total_models):
        playersVelY.append(-9)
        playersAccY.append(1.0)  # Easy mode: gravity 2000 (adjusted for game scale)
        playersFlapped.append(False)
        playersState.append(True)

    alive_players = total_models


    while True:
        for idxPlayer in range(total_models):
            if playersYList[idxPlayer] < 0 and playersState[idxPlayer] == True:
                alive_players -= 1
                playersState[idxPlayer] = False
        if alive_players == 0:
            return {
                'y': 0,
                'groundCrash': True,
                'basex': basex,
                'upperPipes': upperPipes,
                'lowerPipes': lowerPipes,
                'score': score,
                'playerVelY': 0,
            }
        # Update fitness: distance traveled minus distance to closest gap (as per specification)
        frame_count += 1  # Increment frame counter
        for idxPlayer in range(total_models):
            if playersState[idxPlayer] == True:
                # Distance traveled (primary component)
                distance_traveled = frame_count * abs(pipeVelX)
                
                # Distance to closest gap (subtracted from fitness)
                distance_to_gap = 0
                for upper_pipe, lower_pipe in zip(upperPipes, lowerPipes):
                    pipe_x = upper_pipe['x'] + IMAGES['pipe'][0].get_width()
                    if pipe_x > playersXList[idxPlayer]:
                        distance_to_gap = pipe_x - playersXList[idxPlayer]
                        break
                
                # Fitness = distance_traveled - distance_to_closest_gap (as per specification)
                fitness[idxPlayer] = distance_traveled - distance_to_gap
        next_pipe_x += pipeVelX
        for idxPlayer in range(total_models):
            if playersState[idxPlayer] == True:
                pipes_data = {'upper': upperPipes, 'lower': lowerPipes}
                if predict_action(playersYList[idxPlayer], playersXList[idxPlayer], pipes_data, idxPlayer) == 1:
                    if playersYList[idxPlayer] > -2 * IMAGES['player'][0].get_height():
                        playersVelY[idxPlayer] = playerFlapAcc
                        playersFlapped[idxPlayer] = True
                        #SOUNDS['wing'].play()
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            """if (event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP)):
                if playery > -2 * IMAGES['player'][0].get_height():
                    playerVelY = playerFlapAcc
                    playerFlapped = True
                    SOUNDS['wing'].play()
            """

        # check for crash here, returns status list
        crashTest = checkCrash({'x': playersXList, 'y': playersYList, 'index': playerIndex},
                               upperPipes, lowerPipes)

        for idx in range(total_models):
            if playersState[idx] == True and crashTest[idx] == True:
                alive_players -= 1
                playersState[idx] = False
        if alive_players == 0:
            return {
                'y': playery,
                'groundCrash': crashTest[1],
                'basex': basex,
                'upperPipes': upperPipes,
                'lowerPipes': lowerPipes,
                'score': score,
                'playerVelY': 0,
            }

        # check for score - only count each pipe once per frame
        scored_pipes = set()  # Track which pipes have been scored this frame
        for idx in range(total_models):
            if playersState[idx] == True:
                pipe_idx = 0
                playerMidPos = playersXList[idx] + IMAGES['player'][0].get_width() // 2  # Center of bird
                for pipe in upperPipes:
                    pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() // 2  # Center of pipe
                    # Score when bird center passes pipe center
                    if pipeMidPos <= playerMidPos < pipeMidPos + abs(pipeVelX) + 2:
                        # Only increment score if this pipe hasn't been scored by this bird
                        pipe_bird_key = (pipe_idx, idx)
                        if pipe_bird_key not in scored_pipes:
                            # Check if there's a next pipe to set as target
                            if pipe_idx + 1 < len(lowerPipes):
                                next_pipe_x = lowerPipes[pipe_idx+1]['x']
                                next_pipe_hole_y = (lowerPipes[pipe_idx+1]['y'] + (upperPipes[pipe_idx+1]['y'] + IMAGES['pipe'][0].get_height())) / 2
                            individual_scores[idx] += 1  # Individual bird score
                            score = max(individual_scores)  # Global score is best individual score
                            scored_pipes.add(pipe_bird_key)  # Mark this pipe as scored by this bird
                            # SOUNDS['point'].play()
                    pipe_idx += 1

        # playerIndex basex change
        if (loopIter + 1) % 3 == 0:
            playerIndex = next(playerIndexGen)
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 100) % baseShift)

        # player's movement
        for idx in range(total_models):
            if playersState[idx] == True:
                if playersVelY[idx] < playerMaxVelY and not playersFlapped[idx]:
                    playersVelY[idx] += playersAccY[idx]
                if playersFlapped[idx]:
                    playersFlapped[idx] = False
                playerHeight = IMAGES['player'][playerIndex].get_height()
                playersYList[idx] += min(playersVelY[idx], BASEY - playersYList[idx] - playerHeight)

        # move pipes to left
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipe['x'] += pipeVelX
            lPipe['x'] += pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if len(upperPipes) > 0 and 0 < upperPipes[0]['x'] < 5:
            # Calculate position for new pipe based on the last pipe's position
            last_pipe_x = upperPipes[-1]['x']
            newPipe = getRandomPipe(last_pipe_x + (SCREENWIDTH / 2))
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])
        
        # Emergency: if no pipes exist, create new ones
        if len(upperPipes) == 0:
            newPipe = getRandomPipe(SCREENWIDTH + 200)
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if len(upperPipes) > 0 and upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (basex, BASEY))
        # print score so player overlaps the score
        showScore(score)
        for idx in range(total_models):
            if playersState[idx] == True:
                SCREEN.blit(IMAGES['player'][playerIndex], (playersXList[idx], playersYList[idx]))

        pygame.display.update()
        FPSCLOCK.tick(FPS)


def showGameOverScreen(crashInfo):
    """Perform genetic updates here"""
    global current_pool
    global fitness
    global generation
    
    # Get the highest score from crashInfo
    highest_score = crashInfo.get('score', 0)
    
    print(f"\nGeneration {generation} completed!")
    print(f"Best fitness: {max(fitness)}")
    print(f"Average fitness: {sum(fitness)/len(fitness):.2f}")
    print(f"Highest score: {highest_score}")
    
    # Sort models by fitness to identify best performers
    fitness_indices = [(fitness[i], i) for i in range(total_models)]
    fitness_indices.sort(reverse=True)
    
    new_weights = []
    
    # Implementation of the specific replacement strategy:
    # Top 4 birds are retained
    for i in range(4):
        _, idx = fitness_indices[i]
        elite_weights = [w.copy() for w in current_pool[idx].get_weights()]
        new_weights.append(elite_weights)
    
    # 1 offspring from crossover of top 2
    top1_idx = fitness_indices[0][1]
    top2_idx = fitness_indices[1][1]
    offspring1 = model_crossover(top1_idx, top2_idx)[0]
    new_weights.append(model_mutate(offspring1))
    
    # 3 offspring from crossover of random top birds (top 4)
    for _ in range(3):
        parent1_idx = fitness_indices[random.randint(0, 3)][1]
        parent2_idx = fitness_indices[random.randint(0, 3)][1]
        while parent2_idx == parent1_idx:
            parent2_idx = fitness_indices[random.randint(0, 3)][1]
        
        offspring = model_crossover(parent1_idx, parent2_idx)[0]
        new_weights.append(model_mutate(offspring))
    
    # 2 direct replicas of random winners (from top 4)
    for _ in range(2):
        winner_idx = fitness_indices[random.randint(0, 3)][1]
        replica_weights = [w.copy() for w in current_pool[winner_idx].get_weights()]
        new_weights.append(model_mutate(replica_weights))  # Apply mutation to replicas as per spec
    
    # Apply new weights to models
    for i in range(total_models):
        fitness[i] = 0
        current_pool[i].set_weights(new_weights[i])
    
    if save_current_pool == 1:
        save_pool()
    generation = generation + 1
    return

def getRandomPipe(pipeX=None):
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
    gapY += int(BASEY * 0.2)
    pipeHeight = IMAGES['pipe'][0].get_height()
    
    # Use provided x position or default
    if pipeX is None:
        pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE}, # lower pipe
    ]


def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(players, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    statuses = []
    for idx in range(total_models):
        statuses.append(False)

    for idx in range(total_models):
        statuses[idx] = False
        pi = players['index']
        players['w'] = IMAGES['player'][0].get_width()
        players['h'] = IMAGES['player'][0].get_height()
        # if player crashes into ground
        if players['y'][idx] + players['h'] >= BASEY - 1:
            statuses[idx] = True
        playerRect = pygame.Rect(players['x'][idx], players['y'][idx],
                      players['w'], players['h'])
        pipeW = IMAGES['pipe'][0].get_width()
        pipeH = IMAGES['pipe'][0].get_height()

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                statuses[idx] = True
    return statuses

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask

if __name__ == '__main__':
    main()
