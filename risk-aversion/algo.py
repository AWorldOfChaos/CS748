import numpy as np
import math

def MV(mean, var):
    K1 = 0
    K2 = 1
    return K1*mean - K2*var

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError


class MVLCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        
        self.counts = np.zeros(num_arms)
        self.means = np.zeros(num_arms)
        self.vars = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.time = 0
        self.LCB_bounds = np.zeros(num_arms)
        self.delta = 0.1
    
    def give_pull(self):
        return np.argmin(self.LCB_bounds)
    
    def get_reward(self, arm_index, reward):
        self.time += 1
        self.vars[arm_index] = self.counts[arm_index]*(self.vars[arm_index] + (self.means[arm_index] - reward)**2/(self.counts[arm_index] + 1))/(self.counts[arm_index] + 1)
        self.means[arm_index] = (self.counts[arm_index]*self.means[arm_index] + reward)/(self.counts[arm_index] + 1)
        self.counts[arm_index] += 1
        self.values = -1*MV(self.means, self.vars)
        rho = MV(1, 0) / MV(0, 1)
        self.LCB_bounds = self.values - (5 + rho)*np.sqrt(math.log(1/self.delta)/(2*self.counts + 1e-9))


class ExpExp(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        
        self.counts = np.zeros(num_arms)
        self.means = np.zeros(num_arms)
        self.vars = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.time = 0
        self.eps = 0.1
        
    
    def give_pull(self):
        if self.time <= self.eps * self.horizon:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
        
    
    def get_reward(self, arm_index, reward):
        
        self.time += 1
        self.vars[arm_index] = self.counts[arm_index]*(self.vars[arm_index] + (self.means[arm_index] - reward)/(self.counts[arm_index] + 1))/(self.counts[arm_index] + 1)
        self.means[arm_index] = (self.counts[arm_index]*self.means[arm_index] + reward)/(self.counts[arm_index] + 1)
        self.counts[arm_index] += 1
        self.values = MV(self.means, self.vars)