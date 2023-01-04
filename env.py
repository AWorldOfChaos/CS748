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

alg = UCB(num_arms,args.horizon)
total_reward = 0

for i in range(args.horizon):
    index = alg.give_pull()
    reward = arms[index].pull_arm()
    alg.get_reward(index, reward)
    total_reward += reward

print(round(m - total_reward/args.horizon,2))

rho = 0.75
mrho = sorted([x.mean for x in arms])[int(num_arms * rho)]
prob = [1 / num_arms] * num_arms
alg = QRM1(arms, prob, rho, num_arms, args.horizon)
N = 0

total_reward = 0
for i in range(args.horizon):
    # print(i)
    if args.horizon < alg.time:
        break
    reward, pulls = alg.simulate()
    if pulls > 0:
        total_reward += reward / pulls
    N += pulls
# print(N)

print(round(mrho - total_reward / math.log2(N),2))
