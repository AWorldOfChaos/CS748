import argparse
from matplotlib import pyplot as plt
from multiprocessing import Pool
from itertools import repeat
from tqdm import tqdm

from bandit import *
from algo import *
import numpy as np

def run_sim(seed, horizon, algo):
    np.random.seed(seed)
    num_arms = horizon
    arms = []

    m = 0
    for i in range(num_arms):
        arm = BanditArm()
        arms.append(arm)
        m = max(m, arm.mean)

    if algo in ['epsg', 'ucb', 'klucb', 'thompson']:
        
        if algo == 'epsg':
            alg = Eps_Greedy(num_arms,horizon)
        elif algo == 'ucb':
            alg = UCB(num_arms,horizon)
        elif algo == 'klucb':
            alg = KL_UCB(num_arms,horizon)
        elif algo == 'thompson':
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
        N = 0
        
        if algo == 'qrm1':
            alg = QRM1(arms, prob, rho, num_arms, horizon)
        elif algo == 'qrm2':
            alg = QRM2(arms, prob, rho, num_arms, horizon)
        else:
            raise NotImplementedError

        reward = alg.simulate()

        return horizon*mrho - reward

if __name__ == '__main__':
    seeds = np.arange(1000,1100)
    ap = argparse.ArgumentParser()
    ap.add_argument('-H', '--horizon', type = str, help = 'Enter horizons (comma-separated)', required = True)
    ap.add_argument('-A', '--algorithm', type = str, help = 'Enter algorithm name', required = True)
    args = ap.parse_args()
    
    horizon_list = [int(h) for h in args.horizon.strip().split(',')]
    regret_list = []
    for horizon in tqdm(horizon_list):
        with Pool() as pool:
            regret = pool.starmap(run_sim, zip(seeds, repeat(horizon), repeat(args.algorithm)))
        regret_list.append(sum(regret) / len(seeds))
        print(f'Horizon: {horizon} | Regret: {regret_list[-1]}')
    
    plt.title(args.algorithm)
    plt.xlabel('Horizon')
    plt.ylabel('Regret')
    plt.plot(horizon_list,regret_list,'.-',label=args.algorithm)
    plt.savefig(f'figures/{args.algorithm}.png')
    plt.show()