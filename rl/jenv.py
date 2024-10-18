import gymnasium as gym
from gymnasium.spaces import  Box, Discrete
import numpy as np



def generate_task_list(total=1000, seed=None):
    tasks=[]
    rng = np.random.default_rng(seed=seed)
    rr = [
        (int(0.3*total), 10, 100),
        (int(0.45*total), 100, 1000),
        (int(0.25*total), 1000, 10_000),
    ]
    for n, l, h in rr:
        for i in range(n):
            o1, o2 = rng.integers(l,h), rng.integers(l,h)
            tasks.append((f"O{o1}", f"O{o2}", o1, o2))
    return tasks



class JENV(gym.Env):

  @staticmethod
  def johnsons_algorithm(tasks):
    """
    Sorts a list of tasks with two operations using Johnson's algorithm.

    Args:
      tasks: A list of tasks, where each task is a tuple (Oi1, Oi2, di1, di2)
            representing two operations and their durations.

    Returns:
      A new list of tasks sorted in Johnson's order.
    """
    group1 = []
    group2 = []

    # Divide tasks into groups based on operation durations
    for task in tasks:
      if task[2] <= task[3]:  # Compare di1 and di2
        group1.append(task)
      else:
        group2.append(task)

    # Sort groups based on specified criteria
    group1.sort(key=lambda x: x[2])  # Sort by di1 ascending
    group2.sort(key=lambda x: x[3], reverse=True)  # Sort by di2 descending

    # Merge groups into the final sorted list
    sorted_tasks = group1 + group2
    return sorted_tasks

  def __init__(self, n, e, tasks):
    self.n = n
    self.e = e
    self.dim = 4 + self.e

    self.ST = self.johnsons_algorithm(tasks)


    self.observation_space = Box(low=-np.inf, high=np.inf, shape = (self.dim,), dtype=np.float32)
    self.S = self.observation_space.sample()*0
    self.action_space = Discrete(self.e)



  def Umt(self):
    # find Um(t) for which we need uk(t) which is ck(t)/cmax(t)
    # find ck(t) for each executor first
    self.ckt =[max(self.ci1[ei], self.ci2[ei]) for ei in range(self.e)]
    #self.ckt = [ sum(self.E[ei]) for ei in range(self.e) ] # make usre u dont add -ve numbers to ckt
    self.cmaxt = max(self.ckt)

    self.ukt = np.array(self.ckt)*0 if self.cmaxt==0 else np.array(self.ckt)/self.cmaxt


  def reset(self, seed=None, **kwargs):
    super().reset(seed=seed)
    #
    #random.shuffle(self.ST)
    #np.random.default_rng(seed)
    self.t=0  # timestamp init

    oi1, oi2, di1, di2 = self.ST[self.t]
    self.S[:]=0
    self.S[0], self.S[1] = di1, di2

    self.ci1, self.ci2 = [0 for _ in range(self.e)], [0 for _ in range(self.e)]
    #self.prev = [0 for _ in range(self.e)]
    self.ci2[0] =  di1

    """
    Quote:
    It is noted that uk(t) is initialized to 0 at state s0 , which is
    then calculated and ranged from 0 to 1, according to the ratio
    of the completion time of current executor Ek to the makespan
    cmax(t) of the system at s t.
    """
    return self.S, {}

  def step(self, action):


    # task in self.S was placed at executor given by action

    self.ci1[action] += self.S[0]
    self.ci2[action] += self.S[1]
    #ci1_ = self.ci1[action] + self.S[0]
    self.Umt()
    self.t += 1
    done = self.t >= self.n

    if not done:
      c1, c2 = self.ci1[action], self.ci2[action]
      oi1, oi2, di1, di2 = self.ST[self.t]
      self.S[0], self.S[1] = di1, di2
      self.S[2], self.S[3] = c1 , c2
      self.S[4:] = self.ukt

    else:
      self.S[:]=0

    rew = np.mean(self.ukt)*10
    return self.S, rew, done, done, {}




envF = lambda t, e : JENV(len(t), e , t) # training environment