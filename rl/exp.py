# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# exp.py :: explorers and memory for training and testing on gym-like environments
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import torch as tt
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.spaces import Box, Discrete
from pandas import DataFrame

observation_key, action_key, reward_key, done_key, step_key = "O", "A", "R", "D", "T"
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" [A] Static Replay Memory """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class MEM:
    """[MEM] - A key based static replay memory"""

    def __init__(self, capacity, named_spaces, seed) -> None:
        """named_spaces is a dict like string_name vs gym.space"""
        assert capacity > 0
        self.capacity, self.spaces = capacity + 2, named_spaces
        # why add2 to capacity -> one transition requires 2 slots and we need to put an extra slot for current pointer
        self.rng = np.random.default_rng(seed)
        self.build_memory()

    def build_memory(self):
        self.data = {}
        for key, space in self.spaces.items():
            if key != "":
                self.data[key] = np.zeros((self.capacity,) + space.shape, space.dtype)
        self.ranger = np.arange(0, self.capacity, 1)
        self.mask = np.zeros(
            self.capacity, dtype=np.bool8
        )  # <--- should not be changed yet
        self.keys = self.data.keys()
        self.clear()

    def clear(self):
        self.at_max, self.ptr = False, 0
        self.mask *= False

    def length(self):  # Excludes the initial observation (true mask only)
        return len(self.ranger[self.mask])

    def count(self):  # Includes the initial observation (any mask)
        return self.capacity if self.at_max else self.ptr

    def snap(self, mask, **info):
        """snaps all keys in self.keys - assuming info has the keys already (mask=False for first transition after reset)"""
        for k in self.keys:
            self.data[k][self.ptr] = info[k]
        self.mask[self.ptr] = mask
        self.ptr += 1
        if self.ptr == self.capacity:
            self.at_max, self.ptr = True, 0
        self.mask[self.ptr] = False
        return

    def snapk(self, mask, **info):
        """snaps all keys in info - assuming self.keys has the keys already (mask=False for first transition after reset)"""
        for k in info:
            self.data[k][self.ptr] = info[k]
        self.mask[self.ptr] = mask
        self.ptr += 1
        if self.ptr == self.capacity:
            self.at_max, self.ptr = True, 0
        self.mask[self.ptr] = False
        return

    """ NOTE: Sampling

        > sample_methods will only return indices, use self.read(i) to read actual tensors
        > Valid Indices - indices which can be choosen from, indicates which transitions should be considered for sampling
            valid_indices = lambda : self.ranger[self.mask]
    """

    def sample_recent(self, size):
        self.mask[self.ptr] = True
        valid_indices = self.ranger[self.mask]
        self.mask[self.ptr] = False
        iptr = np.where(valid_indices == self.ptr)[0]  # find index of self.ptr in si
        pick = min(len(valid_indices) - 1, size)
        return valid_indices[np.arange(iptr - pick, iptr, 1)]

    def sample_recent_(self, size):
        self.mask[self.ptr] = True
        valid_indices = self.ranger[self.mask]
        self.mask[self.ptr] = False
        iptr = np.where(valid_indices == self.ptr)[0]  # find index of self.ptr in si
        pick = min(len(valid_indices) - 1, size)
        return pick, valid_indices[np.arange(iptr - pick, iptr, 1)]

    def sample_all_(self):
        return self.sample_recent_(self.length())

    def sample_all(self):
        return self.sample_recent(self.length())

    def sample_random(self, size, replace=False):
        valid_indices = self.ranger[self.mask]
        pick = min(len(valid_indices), size)
        return self.rng.choice(valid_indices, pick, replace=replace)

    def sample_random_(self, size, replace=False):
        valid_indices = self.ranger[self.mask]
        pick = min(len(valid_indices), size)
        return pick, self.rng.choice(valid_indices, pick, replace=replace)

    def read(self, i):  # reads [all keys] at [given sample] indices
        return {key: self.data[key][i] for key in self.keys}

    def readkeys(self, i, keys):  # reads [given keys] at [given sample] indices
        return {key: self.data[key][i] for key in keys}

    def readkeis(
        self, ii, keys, teys
    ):  # reads [given keys] at [given sample] indices and rename as [given teys]
        return {t: self.data[k][i] for i, k, t in zip(ii, keys, teys)}

    def readkeist(
        self, *args
    ):  # same as 'readkeis' but the args are tuples like: (index, key, tey)
        return {t: self.data[k][i] for i, k, t in args}

    def prepare_batch(self, batch_size, dtype, device, discrete_action, replace=False):
        pick, samples = self.sample_random_(size=batch_size, replace=replace)
        batch = self.readkeis(
            (samples - 1, samples, samples, samples, samples, samples),
            (
                observation_key,
                observation_key,
                action_key,
                reward_key,
                done_key,
                step_key,
            ),
            ("cS", "nS", "A", "R", "D", "T"),
        )
        # return pick, cS, nS, A, R, D, T
        return (
            pick,
            tt.tensor(batch["cS"], dtype=dtype, device=device),
            tt.tensor(batch["nS"], dtype=dtype, device=device),
            tt.tensor(
                batch["A"], dtype=(tt.long if discrete_action else dtype), device=device
            ),
            tt.tensor(batch["R"], dtype=dtype, device=device),
            tt.tensor(batch["D"], dtype=dtype, device=device),
            tt.tensor(batch["T"], dtype=dtype, device=device),
        )

    """ NOTE: Rendering """

    def render(self, low, high, step=1, p=print):
        p("=-=-=-=-==-=-=-=-=@[MEMORY]=-=-=-=-==-=-=-=-=")
        p(
            "Length:[{}]\tCount:[{}]\nCapacity:[{}]\tPointer:[{}]".format(
                self.length(), self.count(), self.capacity, self.ptr
            )
        )
        for i in range(low, high, step):
            p(
                "____________________________________"
            )  # p_arrow=('<--------[PTR]' if i==self.ptr else "")
            if self.mask[i]:
                p("SLOT: [{}]+".format(i))
            else:
                p("SLOT: [{}]-".format(i))
            for key in self.data:
                p("\t{}: {}".format(key, self.data[key][i]))
        p("=-=-=-=-==-=-=-=-=![MEMORY]=-=-=-=-==-=-=-=-=")

    def render_all(self, p=print):
        self.render(0, self.count(), p=p)

    def render_last(self, nos, p=print):
        self.render(-1, -nos - 1, step=-1, p=p)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" [B] Base Explorer Class """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class EXP:
    """[EXP] - An explorer with memory that can explore an environment using policies
    - remember to set self.pie before exploring
    - setting any policy to None will cause it default to self.random()
    """

    def __init__(self, env) -> None:

        self.spaces = {
            observation_key: env.observation_space,
            action_key: env.action_space,
            reward_key: Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
            done_key: Box(low=0, high=2, shape=(), dtype=np.int32),
            step_key: Box(low=0, high=1, shape=(), dtype=np.int32),
        }

        self.env = env
        self.memory = None
        self.enable_snap()
        self.zero_act, self.zero_reward = (
            self.spaces[action_key].sample() * 0,
            self.spaces[reward_key].sample() * 0,
        )
        # self.reset(clear_memory=False, episodic=episodic)

    def reset(self, clear_memory=False):
        # Note: this does not reset env, just resets its buffer and flag
        self.cs, self.done, self.ts = None, True, 0
        self.act, self.reward = self.zero_act, self.zero_reward
        self.clear_snap() if clear_memory else None
        self.N = 0  # <--- is update after completion of episode or step based on (self.episodic)
        # self.explore = self.explore_episodes if episodic else self.explore_steps

    def enable_snap(self):
        self.snap = self.no_snap if self.memory is None else self.do_snap

    def clear_snap(self):
        if self.memory is not None:
            self.memory.clear()

    def no_snap(self, mask):
        return

    def do_snap(self, mask):
        self.memory.snapk(
            mask,
            **{
                observation_key: self.cs,
                action_key: self.act,
                reward_key: self.reward,
                done_key: self.done,
                step_key: self.ts,
            },
        )

    @tt.no_grad()
    def explore_steps(self, N, reset_seed=None):
        for _ in range(N):
            if self.done:
                self.cs, _ = self.env.reset(seed=reset_seed)
                self.act, self.reward, self.done, self.ts = (
                    self.zero_act,
                    self.zero_reward,
                    False,
                    0,
                )
                self.snap(False)

            self.act = self.get_action()
            self.cs, self.reward, self.done, self.term, _ = self.env.step(self.act)
            self.done = self.done or self.term
            self.ts += 1
            self.snap(True)
            self.N += 1
        return N

    @tt.no_grad()
    def explore_episodes(self, N, reset_seed=None):
        n = 0
        for _ in range(N):
            self.cs, _ = self.env.reset(seed=reset_seed)
            self.act, self.reward, self.done, self.ts = (
                self.zero_act,
                self.zero_reward,
                False,
                0,
            )
            self.snap(False)

            while not self.done:
                self.act = self.get_action()
                self.cs, self.reward, self.done, self.term, _ = self.env.step(self.act)
                self.done = self.done or self.term
                self.ts += 1
                n += 1
                self.snap(True)
            self.N += 1
        return n

    def mode(self, explore_mode="random", pie=None, **kwargs):
        """args:
        explore_mode      pie          argF
        0 random          None         None
        1 policy          obj          None
        2 greedy          tuple        (epsilon, seed)
        3 heuristic       pie          None
        """
        if explore_mode != "random":  # mode not random, requires policy.predict
            if not hasattr(pie, "predict"):
                raise ValueError(f"policy object [{pie}] does not implement predict.")

            self.pie = pie
            if explore_mode == "policy":
                self.get_action = self.get_action_policy
            elif explore_mode == "greedy":
                self.get_action = self.get_action_greedy
                self.epsilon, seed = kwargs.get("epsilon", 0.5), kwargs.get(
                    "seed", None
                )
                self.epsilon_rng = np.random.default_rng(seed)
            elif explore_mode == "heuristic":
                self.get_action = self.get_action_heuristic
                self.epsilon, seed = kwargs.get("epsilon", 0.5), kwargs.get(
                    "seed", None
                )
                self.epsilon_rng = np.random.default_rng(seed)
            elif explore_mode == "heuristicE":
                self.get_action = self.get_action_heuristicE
                self.epsilon, seed = kwargs.get("epsilon", 0.5), kwargs.get(
                    "seed", None
                )
                self.epsilon_rng = np.random.default_rng(seed)
            elif explore_mode == "heuristicC":
                self.get_action = self.get_action_heuristicC
                self.epsilon, seed = kwargs.get("epsilon", 0.5), kwargs.get(
                    "seed", None
                )
                self.epsilon_rng = np.random.default_rng(seed)
        else:  # mode random, do not require policy
            if not (pie is None):
                print(
                    f"Warning: setting policy [{pie}] on a random explorer, this has no effect."
                )
            self.get_action = self.get_action_random

    def get_action_random(self):
        return self.env.action_space.sample()

    def get_action_policy(self):
        return self.pie.predict(self.cs)

    def get_action_greedy(self):
        return (
            self.env.action_space.sample()
            if (self.epsilon_rng.random() < self.epsilon)
            else self.pie.predict(self.cs)
        )

    def get_action_heuristic(self):
        return (
            (
                self.env.action_space.sample()
                if (self.epsilon_rng.random() < 0.25)
                else self.env.heuristic()
            )
            if (self.epsilon_rng.random() < self.epsilon)
            else self.pie.predict(self.cs)
        )

    def get_action_heuristicE(self):
        return (
            (
                self.env.action_space.sample()
                if (self.epsilon_rng.random() < 0.25)
                else self.env.heuristic_edge()
            )
            if (self.epsilon_rng.random() < self.epsilon)
            else self.pie.predict(self.cs)
        )

    def get_action_heuristicC(self):
        return (
            (
                self.env.action_space.sample()
                if (self.epsilon_rng.random() < 0.25)
                else self.env.heuristic_cloud()
            )
            if (self.epsilon_rng.random() < self.epsilon)
            else self.pie.predict(self.cs)
        )


def make_mem(spaces, capacity, seed):
    return MEM(capacity, spaces, seed) if capacity > 0 else None


def make_exp(env, memory_capacity, memory_seed):
    exp = EXP(env)
    mem = make_mem(exp.spaces, memory_capacity, memory_seed)
    exp.memory = mem
    exp.enable_snap()
    return exp, mem


# -----------------------------------------------------------------------------------------------------
""" FOOT NOTE:

"""
# -----------------------------------------------------------------------------------------------------


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" [E] Policy Evaluation/Testing ~ does not use explorers """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@tt.no_grad()
def test_policy(env, pie, max_steps, reset_seed, verbose=0, render=0):
    """test policy for one episode or end of max_steps; returns (steps, return)
    Args:
        env : gym.Env instance
        pie : must implement predict(state)
        max_steps: can be infinity

        render:
                1 = (render final state only)
                2 = (render final and initial state only)
                3 = (render all states)
        verbose:
                1 = start and end only
                2 - details (each step)

    """
    if hasattr(pie, "train"):
        pie.train(False)
    cs, _ = env.reset(seed=reset_seed)
    done, steps = False, 0
    total_reward = 0
    if verbose > 0:
        print("[RESET]")
    if render == 2:
        env.render()

    while not (done or steps >= max_steps):
        steps += 1
        act = pie.predict(cs)
        cs, reward, done, term, _ = env.step(act)
        done = done or term
        total_reward += reward
        if verbose > 1:
            print(
                "[STEP]:[{}], Act:[{}], Reward:[{}], Done:[{}], Return:[{}]".format(
                    steps, act, reward, done, total_reward
                )
            )
        if render == 3:
            env.render()

    if verbose > 0:
        print("[TERM]: Steps:[{}], Return:[{}]".format(steps, total_reward))
    if render == 1 or render == 2:
        env.render()
    return total_reward, steps


def eval_policy(
    env,
    pie,
    max_steps,
    episodes,
    reset_seed,
    verbose=0,
    render=0,
    verbose_result=True,
    render_result=True,
    figsize=(16, 8),
    caption="",
    return_fig=False,
):
    """calls test_policy for multiple episodes;
    returns results as pandas dataframe with cols: (#, steps, return)"""
    test_hist = []
    for n in range(episodes):
        # print(f'\n-------------------------------------------------\n[Test]: {n}\n')
        result = test_policy(
            env, pie, max_steps, reset_seed=reset_seed, verbose=verbose, render=render
        )
        # print(f'steps:[{result[0]}], reward:[{result[1]}]')
        test_hist.append(result)

    test_hist = tt.as_tensor(test_hist)
    test_results = DataFrame(
        data={
            "#": range(len(test_hist)),
            "steps": test_hist[:, 1],
            "return": test_hist[:, 0],
        }
    )
    mean_return = tt.mean(test_hist[:, 0])
    mean_steps = tt.mean(test_hist[:, 1])
    if verbose_result:
        test_rewards = test_results["return"]
        print(
            f"[Test Result]:\n\
        \tTotal-Episodes\t[{episodes}]\n\
        \tMean-Reward\t[{mean_return}]\n\
        \tMedian-Reward\t[{np.median(test_rewards)}]\n\
        \tMax-Reward\t[{np.max(test_rewards)}]\n\
        \tMin-Reward\t[{np.min(test_rewards)}]\n\
        "
        )
        # print(f'\n{test_results.describe()}\n')
    fig_return = (
        plot_test_result(
            test_results, figsize=figsize, caption=caption, return_fig=return_fig
        )
        if render_result
        else None
    )

    return test_results, mean_return, mean_steps, fig_return


def plot_test_result(val_res, figsize, caption, return_fig=False):
    xrange, val_hist_reward, val_hist_steps = (
        val_res["#"],
        val_res["return"],
        val_res["steps"],
    )

    fig, ax = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f"[{caption}]")

    vrax = ax[0, 0]
    # vrax.plot(val_hist_reward, label='return', color='tab:green', linewidth=0.7)
    # vrax.scatter(xrange, val_hist_reward, color='tab:green', marker='.')
    vrax.bar(xrange, val_hist_reward, color="tab:green", label="return")
    vrax.legend()

    hrax = ax[0, 1]
    hrax.hist(val_hist_reward, bins=50, color="tab:green", label="return-dist")
    hrax.legend()

    vsax = ax[1, 0]
    # vsax.plot(val_hist_steps, label='steps', color='tab:purple', linewidth=0.7)
    # vsax.scatter(xrange,  val_hist_steps, color='tab:purple', marker='.')
    vsax.bar(xrange, val_hist_steps, color="tab:blue", label="steps")
    vsax.legend()

    hsax = ax[1, 1]
    hsax.hist(val_hist_steps, bins=50, color="tab:blue", label="steps-dist")
    hsax.legend()

    plt.show()

    return (
        fig if return_fig else None
    )  # plot_validation_result(test_res, figsize, caption, return_fig=return_fig)


def validate_episodes(
    venvs,
    pie,
    episodes,
    max_steps,
    reset_seed,
    episodic_verbose=0,
    episodic_render=0,
    validate_verbose=1,
    validate_render=0,
    validate_figsize=(12, 12),
    validate_caption="validation",
):
    validate_result = []
    for venv in venvs:
        _, mean_return, mean_steps, _ = eval_policy(
            env=venv,
            pie=pie,
            max_steps=max_steps,
            episodes=episodes,
            reset_seed=reset_seed,
            verbose=episodic_verbose,
            render=episodic_render,
            verbose_result=validate_verbose,
            render_result=validate_render,
            figsize=validate_figsize,
            caption=validate_caption,
            return_fig=False,
        )
        # print(test_results.describe())
        validate_result.append((mean_return, mean_steps))
    validate_result = np.array(validate_result)
    return np.mean(validate_result[:, 0]), np.mean(validate_result[:, 1])


def validate_episode(
    venvs, pie, max_steps, reset_seed, episodic_verbose=0, episodic_render=0
):
    validate_result = []
    for venv in venvs:
        reward, steps = test_policy(
            env=venv,
            pie=pie,
            max_steps=max_steps,
            reset_seed=reset_seed,
            verbose=episodic_verbose,
            render=episodic_render,
        )
        validate_result.append((reward, steps))
    validate_result = np.array(validate_result)
    return np.mean(validate_result[:, 0]), np.mean(validate_result[:, 1])


def validate_episode_sum(
    venvs, pie, max_steps, reset_seed, episodic_verbose=0, episodic_render=0
):
    validate_result = []
    for venv in venvs:
        reward, steps = test_policy(
            env=venv,
            pie=pie,
            max_steps=max_steps,
            reset_seed=reset_seed,
            verbose=episodic_verbose,
            render=episodic_render,
        )
        validate_result.append((reward, steps))
    validate_result = np.array(validate_result)
    return (
        np.mean(validate_result[:, 0]),
        np.mean(validate_result[:, 1]),
        np.sum(validate_result[:, 0]),
        np.sum(validate_result[:, 1]),
    )
