# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 12:41:18 2020

@title: AI playing Sonic the hedgehog
@author: Leonardo
@author: Department of Computer Science - University of Oxford - United Kingdom

"""

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
 # IMPORTANT: FIX IT LATER, THIS ROM IS NOT WORKING PROPERLY 





class Worker(object):
    
    def __init__(self, genome, config):
        
        self.genome = genome
        self.config = config
    
    def work(self):
        
        self.env = retro.make(r'SonicTheHedgehog-Genesis',r'GreenHillZone.Act1')
        
        self.env.reset()
        
        ob, _, _, _ = self.env.step(self.env.action_space(self.env.action_space.sample()))

        inx = int(ob.shape[0]/8) 
        iny = int(ob.shape[1]/8)
        
        done = False
        
        net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)
        
        imgarray = []
        
        fitness = 0 
        xpos = 0
        xpos_max = 0
        counter = 0
        
        while not done:
            self.env.render()
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))
            
            imgarray = np.ndarray.flatten(ob)
            
            actions = net.activate(imgarray)
            
            ob, reward, done, info = self.env.step(actions)
            
            xpos = info['x'] # Change each game - Sonic X position
           
            
            if xpos > xpos_max:

                xpos_max = xpos
                counter = 0 
                fitness += 1
                
            else:
                
                counter +=1 
                
            if counter > 250:
                
                done = True
            
            if xpos >= info['screen_x_end']: # End of game

                fitness += 100000
                done = True
            
        print(fitness)
        return fitness
            


def eval_genomes(genome, config):
    
    work_paralell = Worker(genome, config)
    
    return work_paralell.work()                



# Configuring NEAT
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

p = neat.Population(config)

pe = neat.ParallelEvaluator(8, eval_genomes)

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StdOutReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10)) # Every 10 generation uses a checkpoint

winner = p.run(pe.evaluate)

# Save NeuralNet
with open('Sonic_Neat.pkt', 'wb') as output:
    pickle.dump(winner, output, 1)