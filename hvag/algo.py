import numpy as np
import math

def KL(x, y):
    if y>=1:
        return -1

    return (x+0.0001)*math.log((x+0.0001)/(y+0.0001)) + (1.0001-x)*math.log((1.0001-x)/(1.0001-y))

def find_bound(value, time, count):
    target = (math.log(time) + 3*math.log(math.log(time)))/(0.001 + count)
    values = np.arange(value,1,0.01)
    if value > 0.99:
        return 1
    low = 0
    high = len(values) - 1
    mid = 0
 
    while low < high:
        mid = (high + low) // 2

        if KL(value, values[mid]) < target:
            low = mid + 1
        elif KL(value, values[mid]) > target:
            high = mid - 1
        else:
            return values[mid]
    
    return values[low]

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError


class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value


class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.UCB_bounds = np.zeros(num_arms)
        self.time = 0
        
    
    def give_pull(self):
        return np.argmax(self.UCB_bounds)
        
    
    def get_reward(self, arm_index, reward):
        
        self.time += 1
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value
        self.UCB_bounds = [i + math.sqrt(2*math.log(self.time)/ (0.001 + self.counts[index])) for index, i in enumerate(self.values)]
        

class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)

        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.UCB_bounds = np.zeros(num_arms)
        self.time = 1    
    
    def give_pull(self):
        
        return np.argmax(self.UCB_bounds)
        
    def get_reward(self, arm_index, reward):
        
        self.time += 1
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value
        self.UCB_bounds = [find_bound(self.values[i], self.time, self.counts[i]) for i in range(self.num_arms)]


class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        
        self.alphas = np.ones(num_arms)
        self.betas = np.ones(num_arms)
    
    def give_pull(self):
        samples = np.zeros(self.num_arms)

        for i in range(self.num_arms):
            samples[i] = np.random.beta(self.alphas[i], self.betas[i])
        
        return np.argmax(samples)
        
    
    def get_reward(self, arm_index, reward):
        if reward:
            self.alphas[arm_index] += 1
        else:
            self.betas[arm_index] += 1


class MOSS(Algorithm):
    
    def __init__(self, K, horizon):
        self.means = np.zeros(len(K))
        self.counts = np.zeros(len(K))
        self.ucb = np.ones(len(K))
        self.K = K
        self.horizon = horizon
        self.time = 0
    
    def give_pull(self):
        return np.argmax(self.ucb)
    
    def get_reward(self, arm_index, reward):
        self.time += 1
        self.counts[arm_index] += 1
        self.means[arm_index] = (self.means[arm_index] * self.counts[arm_index] + reward) / (self.means[arm_index] + 1)
        self.ucb = self.means + np.sqrt(np.maximum(0, np.log(self.time / (len(self.K) * (self.counts+1e-9))))/(self.counts+1e-9))
    
    def simulate(self):
        total = 0
        for i in range(self.horizon):
            index = self.give_pull()
            reward = self.K[index].pull_arm()
            self.get_reward(index,reward)
            total += reward
        return total


class QRM1(Algorithm):
    
    def __init__(self, A, prob_over_arms, rho, num_arms, horizon):
        super().__init__(num_arms, horizon)
        self.time = 1
        self.A = A
        self.K = []
        self.rho = rho
        self.idx = np.arange(num_arms)
        self.prob = prob_over_arms
        self.horizon = horizon
    
    def simulate(self):
        total_reward = 0
        while(self.horizon>0):
            self.time = min(2*self.time, self.horizon)
            self.horizon -= self.time 
            self.n = math.ceil(1 / self.rho * max(1,0.5 * math.log(self.rho * self.time)))
            self.K = self.A[:self.n]
            M = MOSS(self.K, self.time)
            # reward += M.simulate() - self.time * math.pow((1 - self.rho),self.n)
            reward = M.simulate()
            total_reward += reward
        return reward


class QRM2(Algorithm):

    def __init__(self, A, prob_over_arms, alpha, num_arms, horizon):
        super().__init__(num_arms, horizon)
        self.time = 1
        self.A = A
        self.K = []
        self.alpha = alpha
        self.idx = np.arange(num_arms)
        self.prob = prob_over_arms
        self.horizon = horizon
    
    def simulate(self):
        reward = 0
        while(self.horizon>0):
            self.time = min(2*self.time, self.horizon)
            self.horizon -= self.time 
            self.n = math.ceil(math.pow(self.time,self.alpha))
            self.K = self.A[:self.n]
            M = MOSS(self.K, self.time)
            # reward += M.simulate() - self.time * math.pow((1 - self.rho),self.n)
            reward += M.simulate()
        return reward


def MV(mean, var, rho):
    return var - rho*mean


class MVLCB(Algorithm):
    def __init__(self, num_arms, horizon, rho):
        super().__init__(num_arms, horizon)
        
        self.counts = np.zeros(num_arms)
        self.means = np.zeros(num_arms)
        self.vars = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.time = 0
        self.LCB_bounds = np.zeros(num_arms)
        self.delta = 0.1
        self.rho = rho        
    
    def give_pull(self):
        return np.argmin(self.LCB_bounds)
    
    def get_reward(self, arm_index, reward):
        self.time += 1
        self.vars[arm_index] = self.counts[arm_index]*(self.vars[arm_index] + (self.means[arm_index] - reward)**2/(self.counts[arm_index] + 1))/(self.counts[arm_index] + 1)
        self.means[arm_index] = (self.counts[arm_index]*self.means[arm_index] + reward)/(self.counts[arm_index] + 1)
        self.counts[arm_index] += 1
        self.values = MV(self.means, self.vars, self.rho)
        self.LCB_bounds = self.values - (5 + self.rho)*np.sqrt(math.log(1/self.delta)/(2*self.counts+1e-9))
   

class ExpExp(Algorithm):
    def __init__(self, num_arms, horizon, rho, tau): # tau is Exploration phase
        super().__init__(num_arms, horizon)
        
        self.counts = np.zeros(num_arms)
        self.means = np.zeros(num_arms)
        self.vars = np.zeros(num_arms)
        self.values = np.ones(num_arms)
        self.time = 0
        self.tau = tau
        self.rho = rho
    
    def give_pull(self):
        if self.time <= self.tau:
            return np.random.randint(self.num_arms)
        else:
            return np.argmin(self.values)
    
    def get_reward(self, arm_index, reward):
        
        self.time += 1
        self.vars[arm_index] = self.counts[arm_index]*(self.vars[arm_index] + (self.means[arm_index] - reward)**2/(self.counts[arm_index] + 1))/(self.counts[arm_index] + 1)
        self.means[arm_index] = (self.counts[arm_index]*self.means[arm_index] + reward)/(self.counts[arm_index] + 1)
        self.counts[arm_index] += 1
        self.values = MV(self.means, self.vars, self.rho)
