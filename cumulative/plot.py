import argparse
from matplotlib import pyplot as plt
from multiprocessing import Pool
from itertools import repeat
from tqdm import tqdm

import sys, os
sys.path.append(os.path.abspath('..'))

from bandit import *
from algo import *
import numpy as np

def run_sim(seed, horizon, eta, algo):
    np.random.seed(seed)
    num_arms = int(eta * horizon)
    arms = []

    m = 0
    for i in range(num_arms):
        arm = BanditArm()
        arms.append(arm)
        m = max(m, arm.mean)

    if algo in ['EpsG', 'UCB', 'KLUCB', 'Thompson']:
        if algo == 'EpsG':
            alg = Eps_Greedy(num_arms,horizon)
        elif algo == 'UCB':
            alg = UCB(num_arms,horizon)
        elif algo == 'KLUCB':
            alg = KL_UCB(num_arms,horizon)
        elif algo == 'Thompson':
            alg = Thompson_Sampling(num_arms,horizon)
        total_reward = 0

        for _ in range(horizon):
            index = alg.give_pull()
            reward = arms[index].pull_arm()
            alg.get_reward(index, reward)
            total_reward += reward

        return round(horizon * m - total_reward,2)

    else:
        rho = 0.75
        mrho = sorted([x.mean for x in arms])[int(num_arms * rho)]
        prob = [1 / num_arms] * num_arms

        if algo == 'QRM1':
            alg = QRM1(arms, prob, rho, num_arms, horizon)
        elif algo == 'QRM2':
            alg = QRM2(arms, prob, rho, num_arms, horizon)
        else:
            raise NotImplementedError

        reward = alg.simulate()
        return horizon * mrho - reward

if __name__ == '__main__':
    seeds = np.arange(1000,1100)
    ap = argparse.ArgumentParser()
    ap.add_argument('-H', '--horizon', type = str, help = 'Enter horizons (comma-separated)', required = True)
    ap.add_argument('-e', '--eta', type = str, help = 'Enter ratio of arms to horizon (comma-separated)', required = True)
    ap.add_argument('-N', '--name', type = str, help = 'Enter plot name', required = True)
    ap.add_argument('-A', '--algorithm', type = str, help = 'Enter algorithm name', required = True)
    args = ap.parse_args()
    name = args.name

    horizon_list = [int(h) for h in args.horizon.strip().split(',')]
    eta_list = [float(e) for e in args.eta.strip().split(',')]
    for eta in tqdm(eta_list):
        regret_list = []
        for horizon in tqdm(horizon_list):
            with Pool() as pool:
                regret = pool.starmap(run_sim, zip(seeds, repeat(horizon), repeat(eta), repeat(args.algorithm)))
            regret_list.append(sum(regret) / len(seeds))
            print(f'Horizon: {horizon} | Regret: {regret_list[-1]}')
        plt.xlabel('Horizon')
        plt.ylabel('Mean Regret')
        plt.plot(horizon_list,regret_list,'.-',label=eta)
    plt.title(args.algorithm)
    plt.legend(loc='upper right')
    plt.savefig(f'{name}.png')
    plt.show()
