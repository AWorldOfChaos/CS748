import numpy as np
import math

# class Algorithm:
#     def __init__(self, horizon) -> None:
#         self.num_arms = horizon
#         self.alpha = 0.347
#         self.K = []
#         self.t = 1


#     def pull(self) -> int:
#         # self.t *= 2
#         # n = math.ceil(math.pow(self.t, self.alpha))
#         return np.random.randint(self.num_arms)

#     def update(self, arm_index, reward)-> None:
#         pass


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
    
    def __init__(self, A):
        self.A = A
        self.K = []
        self.N = [1 for _ in range(len(self.A))]
    
    def give_pull(self, time):
        m = np.array([x.mean for x in self.A])[self.K]
        n = np.array(self.N)[self.K]
        z = np.array([0] * len(self.K))
        idx = np.argmax(m + np.sqrt(np.maximum(z,np.log(time / (len(self.K) * n)))))
        return self.K[idx]
    
    def get_reward(self, arm_index, reward):
        n = self.N[arm_index]
        m = self.A[arm_index].mean
        self.N[arm_index] = n + 1
        self.A[arm_index].mean = (m * n + reward) / (n + 1)


class QRM1(Algorithm):
    
    def __init__(self, A, prob_over_arms, rho, num_arms, horizon):
        super().__init__(num_arms, horizon)
        self.time = 1
        self.rho = rho
        self.idx = range(num_arms)
        self.prob = prob_over_arms
        self.M = MOSS(A)
    
    def give_pull(self):
        self.time *= 2
        self.n = math.ceil(1 / self.rho * max(1,0.5 * math.log(self.rho * self.time)))
        addn = int(self.n - len(self.M.K))
        arms = np.random.choice(self.idx,addn,True,self.prob)
        self.M.K.extend(arms)
        return self.M.give_pull(self.time)
    
    def get_reward(self, arm_index, reward):
        self.M.get_reward(arm_index,reward)


class QRM2(Algorithm):
    def __init__(self, A, prob_over_arms, alpha, num_arms, horizon):
        super().__init__(num_arms, horizon)
        self.time = 1
        self.alpha = alpha # 0.347
        self.idx = range(num_arms)
        self.prob = prob_over_arms
        self.M = MOSS(A)
    
    def give_pull(self):
        self.time *= 2
        self.n = math.ceil(math.pow(self.time,self.alpha))
        addn = int(self.n - len(self.K))
        arms = np.random.choice(self.idx,addn,True,self.prob)
        self.M.K.extend(arms)
        return self.M.give_pull(self.time)
    
    def get_reward(self, arm_index, reward):
        self.M.get_reward(arm_index,reward)