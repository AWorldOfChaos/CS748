"""
This environment is to test various algorithms in the cumulative scheme.
Last Modified: 18-01-2023
"""

from bandit import *
from algo import *
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-T', '--horizon', type=int, help="Enter horizon")
parser.add_argument('-S', '--seed', type=int, help="Enter seed", required=False, default=100)
parser.add_argument('-A', '--algo', type=str, help="Enter algorithm", required=False, default='ucb')
args = parser.parse_args()

np.random.seed(args.seed)
num_arms = args.horizon
arms = []

m = 0
for i in range(num_arms):
    arm = BanditArm()
    arms.append(arm)
    m = max(m, arm.mean)

if args.algo in ['epsg', 'ucb', 'klucb', 'thompson']:
    
    if args.algo == 'epsg':
        alg = Eps_Greedy(num_arms,args.horizon)
    elif args.algo == 'ucb':
        alg = UCB(num_arms,args.horizon)
    elif args.algo == 'klucb':
        alg = KL_UCB(num_arms,args.horizon)
    elif args.algo == 'thompson':
        alg = Thompson_Sampling(num_arms,args.horizon)
    total_reward = 0

    for i in range(args.horizon):
        index = alg.give_pull()
        reward = arms[index].pull_arm()
        alg.get_reward(index, reward)
        total_reward += reward

    print(round(m - total_reward/args.horizon,2))
    
else:
    
    rho = 0.75
    mrho = sorted([x.mean for x in arms])[int(num_arms * rho)]
    prob = [1 / num_arms] * num_arms
    alg = QRM2(arms, prob, rho, num_arms, args.horizon)
    N = 0
    
    if args.algo == 'qrm1':
        alg = QRM1(arms, prob, rho, num_arms, args.horizon)
    elif args.algo == 'qrm2':
        alg = QRM2(arms, prob, rho, num_arms, args.horizon)
    else:
        raise NotImplementedError

    total_reward = 0
    for i in range(args.horizon):
        if args.horizon < alg.time:
            break
        reward, pulls = alg.simulate()
        if pulls > 0:
            total_reward += reward
        N += pulls

    print(round(mrho - total_reward / N,2))