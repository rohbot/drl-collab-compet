import os
import random
import numpy as np
from unityagents import UnityEnvironment
import time
import torch
from ppo_agent import Agent

ENV_FILE            = "./Tennis_Linux_NoVis/Tennis.x86_64"
#ENV_FILE            = "./Tennis_Linux/Tennis.x86_64"
WEIGHT_FILE         = "models/ppo_trained.pth"
HIDDEN_SIZE         = 512
  

if __name__ == "__main__":    

    env = UnityEnvironment(file_name=ENV_FILE)
    agent = Agent(env, HIDDEN_SIZE)
    agent.load_weights("models/ppo_trained.pth")
    rewards = []
    for i in range(100):
	    reward = agent.play_episode()
	    rewards.append(reward)
	    average = np.mean(np.array(rewards))
	    print('Episode: %i reward: %.3f average: %.3f ' % (i, reward, average))

