import argparse
from matplotlib import pyplot as plt
import numpy as np
from multiprocessing import Pool
from itertools import repeat
from tqdm import tqdm

import sys, os
sys.path.append(os.path.abspath('..'))

from bandit import *
from algo import *

def run_sim(seed, horizon, eta, algorithm):
    np.random.seed(seed)
    num_arms = int(eta * horizon)
    rho = 0
    arms = GaussianBanditInstance2()
    arms = [GaussianBanditArm() for i in range(num_arms)]
    m = 100
    for arm in arms:
        m = min(m, MV(arm.mean, arm.variance, rho))
        print(arm)

    if algorithm == 'MVLCB':
        alg = MVLCB(num_arms,horizon,rho)
    elif algorithm == 'ExpExp':
        alg = ExpExp(num_arms,horizon,rho,(int)(((horizon/14)**(2/3))*10))
    elif algorithm == 'newAlgo1':
        alg = newAlgo1(num_arms,horizon,rho)

    mean_reward = 0
    count = 0
    var_reward = 0

    for _ in range(horizon):
        index = alg.give_pull()
        reward = arms[index].pull_arm()
        alg.get_reward(index, reward)
        var_reward = count*(var_reward + (mean_reward - reward)**2/(count + 1))/(count + 1)
        mean_reward = (count*mean_reward + reward)/(count + 1)
        print(f'Index: {index} | Reward: {reward}')
        count += 1

    return (MV(mean_reward,var_reward,rho) - m)


print(run_sim(0, 30, 1, 'newAlgo1'))
