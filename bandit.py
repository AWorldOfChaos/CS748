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


class GaussianBanditArm:

    def __init__(self) -> None:
        self.mean = np.random.rand()
        self.variance = np.random.rand() / 16

    def pull_arm(self):
        return np.random.normal(self.mean, math.sqrt(self.variance))


class BanditArm:

    def __init__(self) -> None:
        self.mean = np.random.rand()

    def pull_arm(self):
        return np.random.binomial(1, self.mean)

def GaussianBanditInstance1():
    arms = []
    arm1 = GaussianBanditArm()
    arm1.mean = 1
    arm1.variance = 0.05
    arm2 = GaussianBanditArm()
    arm2.mean = 0.5
    arm2.variance = 0.25
    arms.append(arm1)
    arms.append(arm2)
    return arms

def GaussianBanditInstance2():
    arms = []
    arm1 = GaussianBanditArm()
    arm1.mean = 0.2
    arm1.variance = 0.2
    arm2 = GaussianBanditArm()
    arm2.mean = 0.05
    arm2.variance = 0.05
    arms.append(arm1)
    arms.append(arm2)
    return arms
