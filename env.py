from bandit import *
from algo import *
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-T', '--horizon', type=int, help="Enter horizon")
args = parser.parse_args()

np.random.seed(100)
num_arms = args.horizon
arms = []
m = 0
for i in range(num_arms):
    arm = BanditArm()
    arms.append(arm)
    m = max(m, arm.mean)

alg = Thompson_Sampling(num_arms, args.horizon)
total_reward = 0

for i in range(args.horizon):
    index = alg.give_pull()
    reward = arms[index].pull_arm()
    alg.get_reward(index, reward)
    total_reward += reward

print(round(m - total_reward/args.horizon,2))
