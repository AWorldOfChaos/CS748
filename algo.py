import math
import numpy as np

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
        while self.horizon>0:
            self.time = min(2*self.time, self.horizon)
            self.horizon -= self.time 
            self.n = math.ceil(1 / self.rho * max(1,0.5 * math.log(self.rho * self.time)))
            self.K = np.random.choice(self.A,self.n,self.prob)
            M = MOSS(self.K, self.time)
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
        while self.horizon>0:
            self.time = min(2*self.time, self.horizon)
            self.horizon -= self.time 
            self.n = math.ceil(math.pow(self.time,self.alpha))
            self.K = np.random.choice(self.A,self.n,self.prob)
            M = MOSS(self.K, self.time)
            reward += M.simulate()
        return reward

def MV(mean, var, rho = 1):
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
        self.vars[arm_index] = self.counts[arm_index]*(
            self.vars[arm_index] + (self.means[arm_index] - reward)**2/(self.counts[arm_index] + 1)
        )/(self.counts[arm_index] + 1)
        self.means[arm_index] = (
            self.counts[arm_index]*self.means[arm_index] + reward
        )/(self.counts[arm_index] + 1)
        self.counts[arm_index] += 1
        self.values = MV(self.means, self.vars, self.rho)

class ExpExpSS(Algorithm):

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
        if self.time > self.tau:
            return
        self.time += 1
        self.vars[arm_index] = self.counts[arm_index]*(
            self.vars[arm_index] + (self.means[arm_index] - reward)**2/(self.counts[arm_index] + 1)
        )/(self.counts[arm_index] + 1)
        self.means[arm_index] = (
            self.counts[arm_index]*self.means[arm_index] + reward
        )/(self.counts[arm_index] + 1)
        self.counts[arm_index] += 1
        self.values = MV(self.means, self.vars, self.rho)


class newAlgo0(Algorithm):

    def __init__(self, num_arms, horizon, rho):
        super().__init__(num_arms, horizon)
        self.counts = np.zeros(num_arms)
        self.means = np.zeros(num_arms)
        self.vars = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.time = 0
        self.net_reward = 0
        self.rho = rho
        self.margin = MV(0.99, 0.01, rho)

    def give_pull(self):
        if np.min(self.values) < self.margin:
            # allowed_idx = [i for i in range(self.num_arms) if self.values[i] < self.margin]
            # values2 = [MV(self.means[i] - self.net_reward/(self.time + 1e-5),self.vars[i],self.rho) if i in allowed_idx else 1e10 for i in range(self.num_arms)]
            values2 = MV(self.means - self.net_reward/(self.time + 1e-5),self.vars,self.rho)
            return np.argmin(np.array(values2))
        else:
            return np.random.randint(self.num_arms)

    def get_reward(self, arm_index, reward):
        self.time += 1
        self.vars[arm_index] = self.counts[arm_index]*(
            self.vars[arm_index] + (self.means[arm_index] - reward)**2/(self.counts[arm_index] + 1)
        )/(self.counts[arm_index] + 1)
        self.means[arm_index] = (
            self.counts[arm_index]*self.means[arm_index] + reward
        )/(self.counts[arm_index] + 1)
        self.counts[arm_index] += 1
        self.values = MV(self.means, self.vars, self.rho)
        self.net_reward += reward


class newAlgo1(Algorithm):

    def __init__(self, num_arms, horizon, rho):
        super().__init__(num_arms, horizon)
        self.eta = num_arms/horizon
        self.counts = np.zeros(num_arms)
        self.means = np.zeros(num_arms)
        self.vars = np.zeros(num_arms)
        self.time = 0
        self.net_reward = 0
        self.times = 0
        self.last_pulled = -1
        self.rho = rho
        self.margin = MV(0.90, 0.02, rho)
        self.values = np.ones(self.num_arms)*self.margin

    def give_pull(self):
        if self.times > 0 and self.times < 4:
            self.times += 1
        else:
            self.times = 1
            if np.min(self.values) < self.margin:
                self.last_pulled = np.argmin(np.array(self.values))
            else:
                self.last_pulled = np.random.randint(self.num_arms)
        return self.last_pulled

    def get_reward(self, arm_index, reward):
        self.time += 1
        self.vars[arm_index] = self.counts[arm_index]*(
            self.vars[arm_index] + (self.means[arm_index] - reward)**2/(self.counts[arm_index] + 1)
        )/(self.counts[arm_index] + 1)
        self.means[arm_index] = (
            self.counts[arm_index]*self.means[arm_index] + reward
        )/(self.counts[arm_index] + 1)
        self.counts[arm_index] += 1
        self.values = (MV(self.means, self.vars, self.rho) * (self.counts > 0) +
                       np.ones(self.num_arms)*self.margin * (self.counts == 0))
        self.net_reward += reward


class newAlgo2(Algorithm):

    def __init__(self, num_arms, horizon, rho, eps, quantile):
        super().__init__(num_arms, horizon)
        self.eta = num_arms/horizon
        self.counts = np.zeros(num_arms)
        self.means = np.zeros(num_arms)
        self.vars = np.zeros(num_arms)
        self.time = 0
        self.net_reward = 0
        self.times = 0
        self.last_pulled = -1
        self.rho = rho
        self.eps = eps
        self.T = self.horizon * self.eps
        self.quantile = quantile
        self.margin = MV(10, 0, rho)
        self.values = np.ones(self.num_arms)*self.margin

    def give_pull(self):
        if self.time % (self.horizon // 10) == 0:
            mvList = []
            for mean, var, count in zip(self.means,self.vars,self.counts):
                if count > 0:
                    mvList.append(MV(mean,var,self.rho))
            if int(len(mvList) * self.quantile) < len(mvList):
                self.margin = sorted(mvList, reverse = True)[int(len(mvList) * self.quantile)]
            else:
                if len(mvList) != 0:
                    self.margin = sorted(mvList)[-1]
        if self.times > 0 and self.times < 4:
            self.times += 1
        else:
            self.times = 1
            if (self.time > 0 and self.time < self.T) or (self.time > 0 and np.random.rand() < self.T * self.eps / self.time):
                self.last_pulled = np.random.randint(self.num_arms)
            else:
                if np.min(self.values) < self.margin:
                    self.last_pulled = np.argmin(np.array(self.values))
                else:
                    self.last_pulled = np.random.randint(self.num_arms)
        return self.last_pulled

    def get_reward(self, arm_index, reward):
        self.time += 1
        self.vars[arm_index] = self.counts[arm_index]*(
            self.vars[arm_index] + (self.means[arm_index] - reward)**2/(self.counts[arm_index] + 1)
        )/(self.counts[arm_index] + 1)
        self.means[arm_index] = (
            self.counts[arm_index]*self.means[arm_index] + reward
        )/(self.counts[arm_index] + 1)
        self.counts[arm_index] += 1
        self.values = (MV(self.means, self.vars, self.rho) * (self.counts > 0) +
                       np.ones(self.num_arms)*self.margin * (self.counts == 0))
        self.net_reward += reward
