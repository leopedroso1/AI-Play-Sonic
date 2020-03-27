# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 12:41:18 2020

@title: AI playing Sonic the hedgehog
@author: Leonardo
@author: Department of Computer Science - University of Oxford - United Kingdom

"""

# Idea: Use Q Learning technique to take the best maximum fit!
# Optmization 1: Explore using classes 
# Optmization 2: Use more variables to settle the reward

# IMPORTANT NOTICE: 1º: RETRO: you must install first gym-retro by: pip install gym-retro, 
#                              after installation you need to import your ROMs to Retro folder by python -m retro.import <FOLDER PATH>
#                   2º: NEAT: you must install the NEAT API to use it in your code: pip install neat-python
#                       For more information: https://neat-python.readthedocs.io/en/latest/index.html

import retro
import neat
import cv2
import pickle
import numpy as np


# Loading Sonic environment
env = retro.make(r'SonicTheHedgehog-Genesis',r'GreenHillZone.Act1') # IMPORTANT: FIX IT LATER, THIS ROM IS NOT WORKING PROPERLY 

imgarray = []

xpos_end = 0

def eval_genomes(genomes, config):
    
    for genome_id, genome in genomes:
    
        ob = env.reset() #observation = image to be inputed in our NN
        ac = env.action_space.sample()  # action to be made
        
        # Inputs - Here we can find the parameters for our config file at input shape
        inx, iny, inc = env.observation_space.shape
        
        inx = int(inx/8)
        iny = int(iny/8)
        
        # Creating our Recurrent Neural Network given our genomes and configuration
        net = neat.nn.RecurrentNetwork.create(genome, config)
        
        # Fitness control variables
        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        xpos = 0
        xpos_max = 0 
        
        done = False
        
        while not done:
            
            env.render()
            frame += 1

            # Image treatment             
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))
            
            # Compressing 2D image into an array
            for x in ob:
                for y in x:
                    imgarray.append(y)
            
            # Neural Network output - buttons to be pressed
            nnOutput = net.activate(imgarray)
            
#            print(len(imgarray), nnOutput)


            # Variables description
            # Observation: image on the screen in the time of the action
            # Reward: reward that our model will earn each time (example: score)
            # Done: if the done condition is achieved (Example: lost all lives)
            # Info: dictionary of all values you've set in the data
            ob, rew, done, info = env.step(nnOutput)
            imgarray.clear()
            
            
            # Record xpos make sonic done if he doesn't achieve the goal. 
            # info contains all varabiles from the game. We record x values to track Sonic
            
            xpos = info['x']    
            xpos_end = info['screen_x_end']
            
            
            
            # Tracking Sonic X position
            if xpos > xpos_max:
                
                fitness_current += 1 # Add a reward if Sonic goes to the right direction 
                xpos_max = xpos
            
            if  xpos == xpos_end and xpos > 500:
                fitness_current += 100000 #Maximum established in 'config-feedforward.txt'
                done = True
            
            # the steps above can be abstracted by using the code below
            # fitness_current += rew
            
            
            # Set a counter if achieve a best fitness        
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0

            else:
                counter += 1

            # if loses 3 lives (done) or Sonic doesn't get best fitness (250 chances).
            if done or counter == 250:
                done = True
                print(genome_id, fitness_current) # print results
            
            # Update the genone fitness until not done
            genome.fitness = fitness_current 
                



# Configuring NEAT
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

p = neat.Population(config)

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StdOutReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10)) # Every 10 generation uses a checkpoint

winner = p.run(eval_genomes)


#env.reset() # Reset button

#done = False # If the game is not done we will use to control the loop

#while not done:
    
    # Start the emulator
#    env.render() 
    
    # action = env.action_space.sample() - Test: Generate a sample of controls for testing purposes
    # print(action)
    
    # Actions to be used in the game
 #   action = [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0]


   # Apply the actions and collect the result    
  #  observation, reward, done, info =  env.step(action) # action is a set of buttons from emulator. It returns an array of 0's and 1's. Apply the actions given previously
    
  # print("Action ",action,"Reward", reward)
    