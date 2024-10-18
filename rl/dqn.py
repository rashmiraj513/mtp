""" DQN - Deep Q Networks 

    - DQN can be 
        - Single DQN with no target
        - Single DQN with target
        - Double DQN (uses target as the double)
        - Dueling DQN (architecture changes, updates are same) - use common.DLP
"""

from tqdm import tqdm
import os
from math import inf, nan
import torch as tt
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from .exp import make_exp, validate_episode, validate_episodes
from .modular import clone, requires_grad_


class BaseValueEstimator:
    """
    - Base class for value estimators

    NOTE: For Q-Values, underlying parameters `value_theta` should accept state-action pair as 2 sepreate inputs
    NOTE: All Value functions (V or Q) are called in batch mode only
    """

    def __init__(self, value_theta, discrete_action, has_target, dtype, device):
        self.dtype, self.device = dtype, device
        if discrete_action:
            self.is_discrete = True
            self.call = self.call_discrete
            self.call_ = self.call_discrete_
        else:
            self.is_discrete = False

            self.call = self.call_continuous
            self.call_ = self.call_continuous_
        self.has_target = has_target
        self.theta = value_theta.to(dtype=dtype, device=device)
        self.theta_ = (
            requires_grad_(clone(self.theta), False) if self.has_target else self.theta
        )
        self.parameters = self.theta.parameters

        # Sets to train=False
        self.theta.eval()
        self.theta_.eval()
        self.set_running_params()

    def __call__(self, state, target=False):  # Called in batch mode
        return self.call_(state) if target else self.call(state)

    def update_target(self, polyak=0.0):
        if not self.has_target:
            return False

        if polyak > 0:
            self.update_target_polyak(polyak)
        else:
            self.update_target_state_dict()

        self.theta_.eval()
        return True

    def train(self, mode):
        self.theta.train(mode)
        self.theta_.train(mode)

    def _save(self):
        self.theta.is_discrete = self.is_discrete
        self.theta.has_target = self.has_target
        self.theta.dtype = self.dtype
        self.theta.device = self.device

    def save_(self):
        del (
            self.theta.is_discrete,
            self.theta.has_target,
            self.theta.dtype,
            self.theta.device,
        )

    @staticmethod
    def save(val, path):
        val._save()
        tt.save(val.theta, path)
        val.save_()

    @staticmethod
    def get_running_params(model):
        return [
            param for name, param in model.state_dict().items() if "running_" in name
        ]

    def set_running_params(self):
        self.running_params_base = __class__.get_running_params(self.theta)
        self.running_params_target = (
            __class__.get_running_params(self.theta_) if self.has_target else None
        )

    def update_running_params(self):
        for base_param, target_param in zip(
            self.running_params_base, self.running_params_target
        ):
            target_param.copy_(base_param)

    @tt.no_grad()
    def update_target_polyak(self, tau):
        r"""performs polyak update"""
        for base_param, target_param in zip(
            self.theta.parameters(), self.theta_.parameters()
        ):
            target_param.data.mul_(1 - tau)
            tt.add(target_param.data, base_param.data, alpha=tau, out=target_param.data)
        self.update_running_params()

    @tt.no_grad()
    def update_target_parameters(self):
        r"""copies parameters"""
        for base_param, target_param in zip(
            self.theta.parameters(), self.theta_.parameters()
        ):
            target_param.copy_(base_param)
        self.update_running_params()

    @tt.no_grad()
    def update_target_state_dict(self):
        r"""copies state dict"""
        self.theta_.load_state_dict(self.theta.state_dict())
        self.update_running_params()

    # state-value function, same can be used for multi-Qvalue function based on output of value_theta
    def call_discrete(self, state):  # <- Called in batch mode
        return self.theta(state)

    def call_continuous(self, state):  # <- Called in batch mode
        return self.theta(state)

    def call_discrete_(self, state):  # <- Called in batch mode
        return self.theta_(state)

    def call_continuous_(self, state):  # <- Called in batch mode
        return self.theta_(state)

    def predict(self, state):  # <- Called in explore mode
        # Works for discrete action and multi-QValue only
        state = tt.as_tensor(state, dtype=self.dtype, device=self.device)
        return self.theta(state).argmax().item()

    @staticmethod
    def load(path):
        theta = tt.load(path)

        val = __class__(
            theta, theta.is_discrete, theta.has_target, theta.dtype, theta.device
        )
        val.save_()
        return val


def get_pie(value_theta, has_target, dtype, device):
    return BaseValueEstimator(
        value_theta=value_theta,
        discrete_action=True,
        has_target=has_target,
        dtype=dtype,
        device=device,
    )


def load_pie(path):
    return BaseValueEstimator.load(path)


def save_pie(pie, path):
    BaseValueEstimator.save(pie, path)


def train(
    # policy params [T.P]
    # not required
    # value params [T.V]
    value_theta,
    val_opt,
    value_lrsF,
    value_lrsA,
    # device params [DEV]
    dtype,
    device,
    # env params (training) [E]
    env,
    gamma,
    polyak,
    # learning params [L]
    epochs,
    batch_size,
    # verbf,
    learn_times,
    # explore-exploit [X]
    explore_size,
    epsilonStart,
    epsilonF,
    epsilonSeed,
    reset_seed,
    # memory params [M]
    memory_capacity,
    memory_seed,
    min_memory,
    # validation params [V]
    validations_envs,
    validation_freq,
    validation_max_steps,
    validation_reset_seed,
    validation_episodes,
    validation_verbose,
    validation_render,
    # algorithm-specific params [A]
    double,
    tuf,
    # result params [R]
    plot_results,
    save_at,
    checkpoint_freq,
    explorer,
    clear_memory,
):
    # setup policy
    has_target = double or tuf > 0
    pie = get_pie(value_theta, has_target, dtype, device)
    # pie.train(False)
    # return pie, None, None, None
    # setup explorer and memory
    if explorer is None:
        exp, _ = make_exp(
            env=env,
            memory_capacity=memory_capacity,
            memory_seed=memory_seed,
        )
    else:
        exp = explorer

    # setup optimizer
    # val_opt = value_optF(pie.parameters(), **value_optA)
    val_lrs = value_lrsF(val_opt, **value_lrsA)

    # loss
    val_loss = nn.MSELoss()  # <-- its important that DQN uses MSELoss only

    if save_at:
        os.makedirs(save_at, exist_ok=True)

    do_checkpoint = (checkpoint_freq > 0) and save_at
    # ready training
    do_validate = (
        (len(validations_envs) > 0) and validation_freq and validation_episodes
    )
    mean_validation_return, mean_validation_steps = nan, nan
    validation_max_steps = inf if validation_max_steps is None else validation_max_steps
    train_hist = []
    validation_hist = []
    learn_count, update_count = 0, 0
    exp.reset(clear_memory=clear_memory)  # <-- initially non episodic
    # fill up memory
    len_memory = exp.memory.length()
    if len_memory < min_memory:
        exp.mode("random")
        explored = exp.explore_steps(min_memory - len_memory, reset_seed=reset_seed)
        print(f"[*] Explored Min-Memory [{explored}] Steps")

    exp.reset()
    exp.mode("greedy", pie=pie, epsilon=epsilonStart, seed=epsilonSeed)

    # ------------------------------------pre-training results
    checkpoints = []
    if do_checkpoint:
        check_point_as = os.path.join(save_at, f"pre.pie")
        save_pie(pie, check_point_as)
        print(f"Checkpoint @ {check_point_as}")
        checkpoints.append(check_point_as)

    if do_validate:
        if validation_episodes > 1:
            mean_validation_return, mean_validation_steps = validate_episodes(
                validations_envs,
                pie,
                episodes=validation_episodes,
                max_steps=validation_max_steps,
                reset_seed=validation_reset_seed,
                validate_verbose=validation_verbose,
                validate_render=validation_render,
            )
        else:
            mean_validation_return, mean_validation_steps = validate_episode(
                validations_envs,
                pie,
                max_steps=validation_max_steps,
                reset_seed=validation_reset_seed,
            )
        validation_hist.append((mean_validation_return, mean_validation_steps))
        print(
            f" [Pre-Validation] :: Return:{mean_validation_return}, Steps:{mean_validation_steps}"
        )

    # Pre-training results

    for epoch in tqdm(range(epochs)):
        epoch_ratio = epoch / epochs
        exp.epsilon = epsilonF(epoch_ratio)
        explored = exp.explore_steps(explore_size, reset_seed=reset_seed)

        pie.train(True)

        for _ in range(learn_times):
            #  batch_size, dtype, device, discrete_action, replace=False
            pick, cS, nS, A, R, D, T = exp.memory.prepare_batch(
                batch_size, dtype, device, discrete_action=True
            )
            I = tt.arange(0, pick)
            with tt.no_grad():
                # print(nS.device)
                target_ns = pie(nS, target=True)

                if not double:
                    updater, _ = tt.max(target_ns, dim=1)
                else:
                    _, qmax_ind = tt.max(pie(nS, target=False), dim=1)
                    updater = target_ns[I, qmax_ind[I]]

                q_update = R + gamma * updater * (1 - D)

            pred = pie(cS, target=False)
            with tt.no_grad():
                target = tt.zeros_like(pred)  # pred.detach().clone()
                target.copy_(pred)
            target[I, A[I]] = q_update[I]
            loss = val_loss(pred, target)
            val_opt.zero_grad()
            loss.backward()
            val_opt.step()

        pie.train(False)
        train_hist.append((exp.epsilon, val_lrs.get_last_lr()[-1]))
        val_lrs.step()
        learn_count += 1
        if has_target:
            if learn_count % tuf == 0:
                pie.update_target(polyak=polyak)
                update_count += 1

        if do_checkpoint:
            if (epoch + 1) % checkpoint_freq == 0:
                check_point_as = os.path.join(save_at, f"{epoch+1}.pie")
                save_pie(pie, check_point_as)
                print(f"Checkpoint @ {check_point_as}")
                checkpoints.append(check_point_as)

        if do_validate:
            if (epoch + 1) % validation_freq == 0:
                if validation_episodes > 1:
                    mean_validation_return, mean_validation_steps = validate_episodes(
                        validations_envs,
                        pie,
                        episodes=validation_episodes,
                        max_steps=validation_max_steps,
                        reset_seed=validation_reset_seed,
                        validate_verbose=validation_verbose,
                        validate_render=validation_render,
                    )
                else:
                    mean_validation_return, mean_validation_steps = validate_episode(
                        validations_envs,
                        pie,
                        max_steps=validation_max_steps,
                        reset_seed=validation_reset_seed,
                    )
                validation_hist.append((mean_validation_return, mean_validation_steps))
                print(
                    f" [Validation] :: Return:{mean_validation_return}, Steps:{mean_validation_steps}"
                )
    # @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @
    print(f"[{100:.2f} %]")
    # validate last_time

    if save_at:
        save_as = os.path.join(save_at, f"final.pie")
        save_pie(pie, save_as)
        print(f"Saved @ {save_as}")
        checkpoints.append(save_as)

    if do_validate:
        if validation_episodes > 1:
            mean_validation_return, mean_validation_steps = validate_episodes(
                validations_envs,
                pie,
                episodes=validation_episodes,
                max_steps=validation_max_steps,
                reset_seed=validation_reset_seed,
                validate_verbose=validation_verbose,
                validate_render=validation_render,
            )
        else:
            mean_validation_return, mean_validation_steps = validate_episode(
                validations_envs,
                pie,
                max_steps=validation_max_steps,
                reset_seed=validation_reset_seed,
            )
        validation_hist.append((mean_validation_return, mean_validation_steps))
        print(
            f" [Final-Validation] :: Return:{mean_validation_return}, Steps:{mean_validation_steps}"
        )

    validation_hist, train_hist = np.array(validation_hist), np.array(train_hist)
    if plot_results:
        fig = plot_training_result(validation_hist, train_hist)
    else:
        fig = None

    if save_at:
        save_as = os.path.join(save_at, f"results.npz")
        np.savez(save_as, train=train_hist, val=validation_hist)

    # if do_validate and do_checkpoint:
    #     if len(checkpoints):
    #         best_model = checkpoints[np.argmax(validation_hist[:,0])]
    #     else: best_model=None
    # else:
    #     best_model = None

    return pie, exp, validation_hist, train_hist, fig, checkpoints


def plot_training_result(validation_hist, train_hist):
    tEpsilon, tLR = train_hist[:, 0], train_hist[:, 1]
    vReturn, vSteps = validation_hist[:, 0], validation_hist[:, 1]

    fig, ax = plt.subplots(2, 2, figsize=(16, 6))

    ax_epsilon, ax_lr = ax[0, 1], ax[1, 1]
    ax_return, ax_steps = ax[0, 0], ax[1, 0]

    ax_epsilon.plot(tEpsilon, color="tab:purple", label="Epsilon")
    ax_epsilon.legend()

    ax_lr.plot(tLR, color="tab:orange", label="Learn-Rate")
    ax_lr.legend()

    ax_return.plot(vReturn, color="tab:green", label="Return")
    ax_return.scatter(np.arange(len(vReturn)), vReturn, color="tab:green")
    ax_return.legend()

    ax_steps.plot(vSteps, color="tab:blue", label="Steps")
    ax_steps.scatter(np.arange(len(vSteps)), vSteps, color="tab:blue")
    ax_steps.legend()

    plt.show()
    return fig
