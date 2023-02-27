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
parser.add_argument('-S', '--seed', type=int, help="Enter seed", required = False, default = 100)
args = parser.parse_args()

rho = 0

np.random.seed(args.seed)
num_arms = 2
arms = []
m = -1
for i in range(num_arms):
    arm = GaussianBanditArm()
    if i==0:
        arm.mean = 1
        arm.variance = 0.05
    else:
        arm.mean = 0.5
        arm.variance = 0.25
    arms.append(arm)
    m = max(m, MV(arm.mean, arm.variance, rho))


alg = MVLCB(num_arms,args.horizon)
mean_reward = 0
count = 0
var_reward = 0

for i in range(args.horizon):
    index = alg.give_pull()

    reward = arms[index].pull_arm()
    alg.get_reward(index, reward)

    var_reward = count*(var_reward + (mean_reward - reward)**2/(count + 1))/(count + 1)
    mean_reward = (count*mean_reward + reward)/(count + 1)
    count += 1

print(round((m - MV(mean_reward,var_reward))*args.horizon,2))
