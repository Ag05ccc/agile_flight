#!/usr/bin/env python3
import argparse
import math
#
import os
import subprocess


import numpy as np
import torch
from flightgym import AgileEnv_v1
from ruamel.yaml import YAML, RoundTripDumper, dump
from stable_baselines3.common.utils import get_device
from stable_baselines3.ppo.policies import MlpPolicy


# from rpg_baselines.torch.common.ppo import PPO
from stable_baselines3 import PPO
from rpg_baselines.torch.envs import vec_env_wrapper as wrapper
from rpg_baselines.torch.common.util import test_policy

import tempfile
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env

from imitation.algorithms import bc
from imitation.data import rollout

def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--train", type=int, default=1, help="Train the policy or evaluate the policy")
    parser.add_argument("--render", type=int, default=0, help="Render with Unity")
    parser.add_argument("--trial", type=int, default=1, help="PPO trial number")
    parser.add_argument("--iter", type=int, default=100, help="PPO iter number")
    return parser


def main():
    args = parser().parse_args()

    # load configurations
    cfg = YAML().load(
        open(
            os.environ["FLIGHTMARE_PATH"] + "/flightpy/configs/vision/config.yaml", "r"
        )
    )

    if not args.train:
        cfg["simulation"]["num_envs"] = 1 


    print(cfg["environment"]["goal_vel"])

    # 1- Training environment
    train_env = AgileEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    train_env = wrapper.FlightEnvVec(train_env)

    # set random seed
    configure_random_seed(args.seed, env=train_env)
    # configure_random_seed(0, env=train_env)

    if args.render:
        cfg["unity"]["render"] = "yes"
    
    # create evaluation environment
    old_num_envs = cfg["simulation"]["num_envs"]
    cfg["simulation"]["num_envs"] = 1
    eval_env = wrapper.FlightEnvVec(
        AgileEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    )
    cfg["simulation"]["num_envs"] = old_num_envs

    # save the configuration and other files
    rsg_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = rsg_root + "/../saved"
    os.makedirs(log_dir, exist_ok=True)

    # 2 Expert Olustur
    if args.train:
        
        # expert_first = PPO(
        #     tensorboard_log=log_dir,
        #     policy=MlpPolicy,
        #     policy_kwargs=dict(
        #         activation_fn=torch.nn.ReLU,
        #         net_arch=dict(pi=[256, 256], vf=[512, 512]),
        #         # net_arch=[dict(pi=[64, 64], vf=[64, 64])],
        #         log_std_init=-0.5,
        #     ),
        #     env=train_env,
        #     # eval_env=eval_env,
        #     # use_tanh_act=True,
        #     gae_lambda=0.95,
        #     gamma=0.99,
        #     n_steps=250,
        #     ent_coef=0.0,
        #     vf_coef=0.5,
        #     max_grad_norm=0.5,
        #     batch_size=25000,
        #     clip_range=0.2,
        #     use_sde=False,  # don't use (gSDE), doesn't work
        #     # env_cfg=cfg,
        #     verbose=1,
        #     device="cuda",
        # )

        
        # # print(expert.predict(train_env.reset()))

        # # model.learn(total_timesteps=int(5 * 1e7), log_interval=(10, 50))
        # expert_first.learn(total_timesteps=int(1 * 1e7), log_interval=10)
        # expert_first.save("ppo_expert")

        print("------------- TRAIN DONE ---------------")
        print("------------- TRAIN DONE ---------------")
        print("------------- TRAIN DONE ---------------")
        print("------------- TRAIN DONE ---------------")

        ##--------------------- LOAD -----------------------
        # expert = PPO.load("ppo_expert", env=train_env, device="cuda")
        weight = rsg_root + "/../saved/PPO_{0}/Policy/iter_{1:05d}.pth".format(args.trial, args.iter)
        env_rms = rsg_root +"/../saved/PPO_{0}/RMS/iter_{1:05d}.npz".format(args.trial, args.iter)
        saved_variables = torch.load(weight, map_location="cuda")
        
        # Create policy object
        expert = MlpPolicy(**saved_variables["data"])
        #
        # expert.action_net = torch.nn.Sequential(expert.action_net, torch.nn.Tanh())
        # Load weights
        expert.load_state_dict(saved_variables["state_dict"], strict=False)
        expert.to("cuda")
        # 
        eval_env.load_rms(env_rms)

        # --------------------- BC TRAINING -----------------------
        # --------------------- BC TRAINING -----------------------
        expert_reward, _ = evaluate_policy(expert, eval_env, n_eval_episodes=10)
        print(f"Expert Policy Reward: {expert_reward}")
        print(f"Expert Policy Reward: {expert_reward}")
        print(f"Expert Policy Reward: {expert_reward}")

        # 3- Rollout
        rng = np.random.default_rng()
        rollouts = rollout.rollout(
            expert,
            train_env,
            rollout.make_sample_until(min_timesteps=None, min_episodes=50),
            rng=rng,
            unwrap=False,
        )

        transitions = rollout.flatten_trajectories(rollouts)
        bc_trainer = bc.BC(
            observation_space=train_env.observation_space,
            action_space=train_env.action_space,
            demonstrations=transitions,
            rng=rng,
            device="cuda",
        )

        reward_before_training, _ = evaluate_policy(bc_trainer.policy, eval_env, 10)
        print(f"Reward before training: {reward_before_training}")
        # BURASI CALISMIYOR - REWARD ARTMIYOR
        bc_trainer.train(n_epochs=100)
        print("------------- BC TRAINING DONE ---------------")
        reward_after_training, _ = evaluate_policy(bc_trainer.policy, eval_env, 10)
        print(f"Reward before training: {reward_before_training}")
        print(f"Reward after training: {reward_after_training}")
        print(f"Expert Policy Reward: {expert_reward}")
        # --------------------- BC TRAINING END -----------------------
        # --------------------- BC TRAINING END -----------------------
        bc_trainer.policy.save("bc_policy")

    else:
        # python3 -m run_imitation_bc --render 0 --train 0 --trial 61 --iter 1700 
        # python3 -m run_imitation_bc --render 1 --train 0 --trial 61 --iter 1700 
        if args.render and False:
            proc = subprocess.Popen(os.environ["FLIGHTMARE_PATH"] + "/flightrender/RPG_Flightmare.x86_64")
        
        # LOAD POLICY
        weight = rsg_root + "/../saved/PPO_{0}/Policy/iter_{1:05d}.pth".format(args.trial, args.iter)
        env_rms = rsg_root +"/../saved/PPO_{0}/RMS/iter_{1:05d}.npz".format(args.trial, args.iter)
        print("WEGIHT PATH : ", weight)
        print("WEGIHT PATH : ", weight)
        print("WEGIHT PATH : ", weight)
        device = get_device("auto")
        saved_variables = torch.load(weight, map_location=device)
        
        # Create policy object
        expert = MlpPolicy(**saved_variables["data"])
        #
        # expert.action_net = torch.nn.Sequential(expert.action_net, torch.nn.Tanh())
        # Load weights
        expert.load_state_dict(saved_variables["state_dict"], strict=False)
        expert.to(device)
        # 
        eval_env.load_rms(env_rms)
        test_policy(eval_env, expert, render=args.render)

        # Alternatif model yukleme yontemi
        # expert = PPO.load("ppo_expert", env=train_env, device="cuda")
        
        # --------------------- BC TRAINING -----------------------


        if args.render and False:
            proc.terminate()


if __name__ == "__main__":
    main()
