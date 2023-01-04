"""
This environment is to test various algorithms in the risk aversion scheme.
Last Modified: 04-01-2023
"""


from bandit import *
from algo import *
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('-T', '--horizon', type=int, help="Enter horizon")
parser.add_argument('-S', '--seed', type=int, help="Enter seed")
args = parser.parse_args()

np.random.seed(args.seed)
num_arms = args.horizon
arms = []
m = -1
for i in range(num_arms):
    arm = GaussianBanditArm()
    arms.append(arm)
    m = max(m, MV(arm.mean, arm.variance))

alg = ExpExp(num_arms,args.horizon)
mean_reward = 0
count = 0
var_reward = 0

for i in range(args.horizon):
    index = alg.give_pull()
    reward = arms[index].pull_arm()
    alg.get_reward(index, reward)

    var_reward = count*(var_reward + (mean_reward - reward)/(count + 1))/(count + 1)
    mean_reward = (count*mean_reward + reward)/(count + 1)
    count += 1


print(round(m - MV(mean_reward,var_reward),2))
