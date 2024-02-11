"""This is a simple example demonstrating how to clone the behavior of an expert.

Refer to the jupyter notebooks for more detailed examples of how to use the algorithms.
"""
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env



from imitation.policies.serialize import policy_registry
from stable_baselines3.common import policies


import numpy as np
from imitation.policies.serialize import load_policy
from imitation.util import util

import os
from flightgym import AgileEnv_v1
from rpg_baselines.torch.envs import vec_env_wrapper as wrapper
from ruamel.yaml import YAML, RoundTripDumper, dump
import torch


cfg = YAML().load(
    open(
        os.environ["FLIGHTMARE_PATH"] + "/flightpy/configs/vision/config.yaml", "r"
    )
)
def my_policy_loader(venv=None, some_param=None) -> policies.BasePolicy:
    # load your policy here
        # load configurations

    train_env = AgileEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    train_env = wrapper.FlightEnvVec(train_env)
    # local_policy = load_policy("ppo", train_env, path="/home/gazi13/catkin_ws_agile/src/agile_flight/envtest/saved/PPO_64a/Policy/iter_02000.pth")

    weight = "/home/gazi13/catkin_ws_agile/src/agile_flight/envtest/saved/PPO_4/Policy/iter_02000.pth"
    env_rms = "/home/gazi13/catkin_ws_agile/src/agile_flight/envtest/saved/PPO_4/RMS/iter_02000.npz"

    
    saved_variables = torch.load(weight, map_location="cuda")
    # Create policy object
    local_policy = MlpPolicy(**saved_variables["data"])

    return local_policy

# policy_registry.register("ppo", my_policy_loader)



rng = np.random.default_rng(0)
# env = make_vec_env(
#     "seals:seals/CartPole-v0",
#     rng=rng,
#     post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
# )

student_env = AgileEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
student_env = wrapper.FlightEnvVec(student_env)


def train_expert():
    # note: use `download_expert` instead to download a pretrained, competent expert
    print("Training a expert.")
    expert = PPO(
        policy=MlpPolicy,
        env=student_env,
        seed=0,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=10,
        n_steps=64,
    )
    expert.learn(1_000)  # Note: change this to 100_000 to train a decent expert.
    return expert


def download_expert():
    print("Downloading a pretrained expert.")
    expert = load_policy(
        "ppo-huggingface",
        organization="HumanCompatibleAI",
        env_name="sac-seals-Humanoid-v0",
        venv=student_env,
    )
    return expert





def sample_expert_transitions():
    # expert = train_expert()  # uncomment to train your own expert
    # expert = download_expert()
    expert = my_policy_loader()

    print("Sampling expert transitions.")
    rollouts = rollout.rollout(
        expert,
        student_env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=50),
        rng=rng,
    )
    return rollout.flatten_trajectories(rollouts)


transitions = sample_expert_transitions()
bc_trainer = bc.BC(
    observation_space=student_env.observation_space,
    action_space=student_env.action_space,
    demonstrations=transitions,
    rng=rng,
)

evaluation_env = make_vec_env(
    "seals:seals/CartPole-v0",
    rng=rng,
    env_make_kwargs={"render_mode": "human"},  # for rendering
)

print("Evaluating the untrained policy.")
reward, _ = evaluate_policy(
    bc_trainer.policy,  # type: ignore[arg-type]
    evaluation_env,
    n_eval_episodes=3,
    render=True,  # comment out to speed up
)
print(f"Reward before training: {reward}")

print("Training a policy using Behavior Cloning")
bc_trainer.train(n_epochs=1)

print("Evaluating the trained policy.")
reward, _ = evaluate_policy(
    bc_trainer.policy,  # type: ignore[arg-type]
    evaluation_env,
    n_eval_episodes=3,
    render=True,  # comment out to speed up
)
print(f"Reward after training: {reward}")