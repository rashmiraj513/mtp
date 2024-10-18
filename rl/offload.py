import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np


def generate_task_list(total=1000, seed=None):
    tasks = []
    rng = np.random.default_rng(seed=seed)
    rr = [
        # (int(0.3*total), 10, 100),
        (int(1.0 * total), 10, 100),
        # (int(0.25*total), 1000, 10_000),
    ]
    for n, l, h in rr:
        for i in range(n):
            o1, o2 = rng.integers(l, h), rng.integers(l, h)
            tasks.append((f"O{o1}", f"O{o2}", o1, o2))
    return tasks


class OFFENV(gym.Env):

    def __init__(self, e, tasks):
        self.n = len(tasks)
        self.e = e
        self.dim = 4 + self.e
        self.ST = tasks
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.dim,), dtype=np.float32
        )
        self.S = self.observation_space.sample() * 0
        self.action_space = Discrete(self.e)

    def Umt(self):
        self.ckt = [max(self.ci1[ei], self.ci2[ei]) for ei in range(self.e)]
        self.cmaxt = max(self.ckt)
        self.ukt = (
            np.array(self.ckt) * 0
            if self.cmaxt == 0
            else np.array(self.ckt) / self.cmaxt
        )

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self.t = 0  # timestamp init
        oi1, oi2, di1, di2 = self.ST[self.t]
        self.S[:] = 0
        self.S[0], self.S[1] = di1, di2
        self.ci1, self.ci2 = [0 for _ in range(self.e)], [0 for _ in range(self.e)]
        self.ci2[0] = di1
        return self.S, {}

    def step(self, action):
        # task in self.S was placed at executor given by action
        self.ci1[action] += self.S[0]
        self.ci2[action] += self.S[1]
        self.Umt()
        self.t += 1
        done = self.t >= self.n
        if not done:
            c1, c2 = self.ci1[action], self.ci2[action]
            oi1, oi2, di1, di2 = self.ST[self.t]
            self.S[0], self.S[1] = di1, di2
            self.S[2], self.S[3] = c1, c2
            self.S[4:] = self.ukt
        else:
            self.S[:] = 0

        rew = np.mean(self.ukt) * 10
        return self.S, rew, done, done, {}

    def render(self):
        print(f"TASKS: {self.ST[0:4]} :: Executors: {self.ST[4:]}")
        return
