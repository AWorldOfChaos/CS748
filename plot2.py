import argparse
from matplotlib import pyplot as plt
import numpy as np
from multiprocessing import Pool
from itertools import repeat
from tqdm import tqdm

from bandit import *
from algo import *
import argparse

def run_sim(seed, horizon, algorithm):
    np.random.seed(seed)
    num_arms = 2
    rho = 0
    arms = []
    m = 100
    for i in range(num_arms):
        arm = GaussianBanditArm()
        if i==0:
            arm.mean = 1
            arm.variance = 0.05
        else:
            arm.mean = 0.5
            arm.variance = 0.25
        arms.append(arm)
        m = min(m, MV(arm.mean, arm.variance, rho))

    if algorithm == 'MVLCB':
        alg = MVLCB(num_arms,horizon,rho)
    elif algorithm == 'ExpExp':
        alg = ExpExp(num_arms,horizon,rho)
    mean_reward = 0
    count = 0
    var_reward = 0

    for i in range(horizon):
        index = alg.give_pull()

        reward = arms[index].pull_arm()
        alg.get_reward(index, reward)

        var_reward = count*(var_reward + (mean_reward - reward)**2/(count + 1))/(count + 1)
        mean_reward = (count*mean_reward + reward)/(count + 1)
        count += 1

    return (MV(mean_reward,var_reward,rho)-m)


if __name__ == '__main__':
    seeds = np.arange(0,100)
    ap = argparse.ArgumentParser()
    ap.add_argument('-H', '--horizon', type = str, help = 'Enter horizons (comma-separated)', required = True)
    ap.add_argument('-N', '--name', type = str, help = 'Enter plot name', required = True)
    ap.add_argument('-A', '--algorithm', type = str, help = 'Enter algorithm name', required = True)
    args = ap.parse_args()
    name = args.name

    horizon_list = [int(h) for h in args.horizon.strip().split(',')]
    regret_list = []
    for horizon in tqdm(horizon_list):
        with Pool() as pool:
            regret = pool.starmap(run_sim, zip(seeds, repeat(horizon), repeat(args.algorithm)))
        regret_list.append(sum(regret) / len(seeds))
        print(f'Horizon: {horizon} | Regret: {regret_list[-1]}')

    plt.title(args.algorithm)
    plt.xlabel('Horizon')
    plt.ylabel('Mean Regret')
    plt.plot(horizon_list,regret_list)
    plt.savefig(f'{name}.png')
    plt.show()