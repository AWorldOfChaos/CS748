import numpy as np
import math

class UniformBanditArm:
    def __init__(self) -> None:
        self.mean = np.random.rand()
        self.variance = np.random.rand()
        self.b = self.mean + math.sqrt(self.variance*12)/2
        self.a = self.mean - math.sqrt(self.variance*12)/2
    
    def pull_arm(self):
        return np.random.uniform(low=self.a, high=self.b)


class GaussiamBanditArm:
    def __init__(self) -> None:
        self.mean = np.random.rand()
        self.variance = np.random.rand()
    
    def pull_arm(self):
        return np.random.normal(self.mean, math.sqrt(self.variance))


class BanditArm:
    def __init__(self) -> None:
        self.mean = np.random.rand()
    
    def pull_arm(self):
        return np.random.binomial(1, self.mean)