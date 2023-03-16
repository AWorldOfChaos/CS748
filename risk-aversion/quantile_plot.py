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
    arms = [GaussianBanditArm() for i in range(num_arms)]
    quantile = 0.1
    m = sorted([MV(arm.mean,arm.variance,rho) for arm in arms])[int(num_arms * quantile)]

    if algorithm == 'MVLCB':
        alg = MVLCB(num_arms,horizon,rho)
    elif algorithm == 'ExpExp':
        alg = ExpExp(num_arms,horizon,rho,(int)(((horizon/14)**(2/3))*10))
    elif algorithm == 'ExpExpSS':
        C = 1
        alpha = 0.3
        # horizon = 2 * ((C/quantile)**3 / 196)**(1/alpha)
        num_arms = int(C/quantile * (horizon**(1/3-alpha))+1)
        alg = ExpExp(num_arms,horizon,rho,(int)(((horizon/14)**(2/3))*10))
    elif algorithm == 'newAlgo0':
        alg = newAlgo0(num_arms,horizon,rho)
    elif algorithm == 'newAlgo1':
        alg = newAlgo1(num_arms,horizon,rho)
    elif algorithm == 'newAlgo2':
        eps = 0.1
        alg = newAlgo2(num_arms,horizon,rho,eps,quantile)

    mean_reward = 0
    count = 0
    var_reward = 0

    for _ in range(horizon):
        index = alg.give_pull()
        reward = arms[index].pull_arm()
        alg.get_reward(index, reward)
        var_reward = count*(var_reward + (mean_reward - reward)**2/(count + 1))/(count + 1)
        mean_reward = (count*mean_reward + reward)/(count + 1)
        count += 1

    return (MV(mean_reward,var_reward,rho) - m)

if __name__ == '__main__':
    seeds = np.arange(100,350)
    ap = argparse.ArgumentParser()
    ap.add_argument('-H', '--horizon', type = str, help = 'Enter horizons (comma-separated)', required = True)
    ap.add_argument('-e', '--eta', type = str, help = 'Enter ratio of arms to horizon (comma-separated)', required = True)
    ap.add_argument('-N', '--name', type = str, help = 'Enter plot name', required = True)
    ap.add_argument('-A', '--algorithm', type = str, help = 'Enter algorithm name', required = True)
    args = ap.parse_args()
    name = args.name

    horizon_list = [int(h) for h in args.horizon.strip().split(',')]
    eta_list = [float(e) for e in args.eta.strip().split(',')]
    for eta in eta_list:
        regret_list = []
        for horizon in horizon_list:
            with Pool() as pool:
                regret = pool.starmap(run_sim, zip(seeds, repeat(horizon), repeat(eta), repeat(args.algorithm)))
            regret_list.append(sum(regret) / len(seeds))
            print(f'Horizon: {horizon} | Regret: {regret_list[-1]}')
        plt.xlabel('Horizon')
        plt.ylabel('Quantile Regret')
        # plt.plot(horizon_list,regret_list,'.-',label=f'eta={eta}')
        plt.plot(horizon_list,regret_list,'.-',label=args.algorithm)

    for eta in eta_list:
        regret_list = []
        for horizon in horizon_list:
            with Pool() as pool:
                regret = pool.starmap(run_sim, zip(seeds, repeat(horizon), repeat(eta), repeat('newAlgo2')))
            regret_list.append(sum(regret) / len(seeds))
            print(f'Horizon: {horizon} | Regret: {regret_list[-1]}')
        plt.xlabel('Horizon')
        plt.ylabel('Quantile Regret')
        # plt.plot(horizon_list,regret_list,'.-',label=f'eta={eta}')
        plt.plot(horizon_list,regret_list,'.-',label='newAlgo2')

    for eta in eta_list:
        regret_list = []
        for horizon in horizon_list:
            with Pool() as pool:
                regret = pool.starmap(run_sim, zip(seeds, repeat(horizon), repeat(eta), repeat('ExpExp')))
            regret_list.append(sum(regret) / len(seeds))
            print(f'Horizon: {horizon} | Regret: {regret_list[-1]}')
        plt.xlabel('Horizon')
        plt.ylabel('Quantile Regret')
        plt.plot(horizon_list,regret_list,'.-',label='ExpExp')

    for eta in eta_list:
        regret_list = []
        for horizon in horizon_list:
            with Pool() as pool:
                regret = pool.starmap(run_sim, zip(seeds, repeat(horizon), repeat(eta), repeat('MVLCB')))
            regret_list.append(sum(regret) / len(seeds))
            print(f'Horizon: {horizon} | Regret: {regret_list[-1]}')
        plt.xlabel('Horizon')
        plt.ylabel('Quantile Regret')
        plt.plot(horizon_list,regret_list,'.-',label='MVLCB')
    plt.title('Risk Averse: ' + args.algorithm + ' vs ExpExp vs MVLCB')
    # plt.title('Risk Averse: ' + args.algorithm)

    # horizon_list = [int(h) for h in args.horizon.strip().split(',')]
    # eta_list = [float(e) for e in args.eta.strip().split(',')]
    # for eta in tqdm(eta_list):
    #     regret_list = []
    #     for horizon in tqdm(horizon_list):
    #         with Pool() as pool:
    #             regret = pool.starmap(run_sim, zip(seeds, repeat(horizon), repeat(eta), repeat(args.algorithm)))
    #         regret_list.append(sum(regret) / len(seeds))
    #         print(f'Horizon: {horizon} | Regret: {regret_list[-1]}')
    #     plt.xlabel('Horizon')
    #     plt.ylabel('Mean Regret')
    #     plt.plot(horizon_list,regret_list,'.-',label=eta)
    # plt.title('Risk Averse: ' + args.algorithm)

    plt.legend(loc='upper right')
    plt.savefig(f'figures/{name}.png')
    plt.show()
