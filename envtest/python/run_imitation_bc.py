#!/usr/bin/env python3
import argparse
import math
#
import os
import subprocess
import time

import numpy as np
import torch
from flightgym import AgileEnv_v1
# from flightgym import StudentEnv_v1
from ruamel.yaml import YAML, RoundTripDumper, dump
from stable_baselines3.common.utils import get_device
from stable_baselines3.ppo.policies import MlpPolicy, CnnPolicy, MultiInputPolicy
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
from imitation.data.types import DictObs
import torch as th

from typing import Callable


def configure_random_seed(seed, env=None):
    print("seed : ",seed)
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--train", type=int, default=1, help="Train the policy or evaluate the policy")
    parser.add_argument("--teach", type=int, default=0, help="Teach the policy with imitation learning")
    parser.add_argument("--test", type=int, default=0, help="Test the policy")
    parser.add_argument("--render", type=int, default=0, help="Render with Unity")
    parser.add_argument("--trial", type=int, default=1, help="PPO trial number")
    parser.add_argument("--iter", type=int, default=100, help="PPO iter number")
    parser.add_argument("--imitation", type=int, default=0, help="Imitation or Teacher Policy")
    return parser

def save_mode_custom(name: str, policy):
    
    rsg_root = os.path.dirname(os.path.abspath(__file__))
    policy_path = rsg_root + "/../save_imitation/policy_imitation/"
    os.makedirs(policy_path, exist_ok=True)
    policy.save(policy_path + "/"+ name + ".pth")
    print("BC Trained Policy Saved ...  ",policy_path)
    # self.venv.save_rms(
    #     save_dir=self.logger.get_dir() + "/RMS_imitation", n_iter=batch_num
    # )

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value:
    :return: current learning rate depending on remaining progress
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func

def main():
    args = parser().parse_args()

    # Save the configuration and other files
    rsg_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = rsg_root + "/../saved"
    os.makedirs(log_dir, exist_ok=True)
    w_path = rsg_root + "/../saved/PPO_{0}/Policy/iter_{1:05d}.pth".format(args.trial, args.iter)
    
    if not os.path.exists(w_path):
        print(" TRAIN MODEL ")
        args.train = 1
        args.teach = 0
        args.test = 0
        args.render = 0
        args.imitation = 0
    else:
        print(" TEST MODEL ")
        args.train = 0
        args.teach = 0
        args.test = 1
        args.render = 1
        args.imitation = 1
        
    # Load configurations
    cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] + "/flightpy/configs/vision/config.yaml", "r"))
    
    # Change config according to args
    if not args.train and not args.teach:
        cfg["simulation"]["num_envs"] = 1
    else:
        cfg["simulation"]["num_envs"] = 300 #100
        cfg["simulation"]["num_threads"] = 20
    if args.render:
        cfg["unity"]["render"] = "yes"

    # DEBUG - PERFORMANS
    # cfg["simulation"]["num_threads"] = 10
    # cfg["simulation"]["sim_dt"] = 0.03
    # cfg["simulation"]["max_t"] = 20.0
    cfg["rgb_camera"]["width"] = 800 # 160
    cfg["rgb_camera"]["height"] = 600 # 80

    # Training environment
    cfg["environment"]["student_flag"] = False
    train_env = AgileEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    train_env = wrapper.FlightEnvVec(train_env)
    print("Train environment defined ...")

    # Set random seed
    configure_random_seed(args.seed, env=train_env)

    # Evaluation environment
    old_num_envs = cfg["simulation"]["num_envs"]
    cfg["simulation"]["num_envs"] = 1
    cfg["environment"]["student_flag"] = True # FARKLI
    eval_env = wrapper.FlightEnvVec(AgileEnv_v1(dump(cfg, Dumper=RoundTripDumper), False))
    cfg["simulation"]["num_envs"] = old_num_envs
    
    print("Evaluation environment defined ...")
    
    
    # Open Unity
    if args.render and args.train:
        print("Start Unity process ...")
        train_proc = subprocess.Popen(os.environ["FLIGHTMARE_PATH"] + "/flightrender/RPG_Flightmare.x86_64")
        time.sleep(1)
        train_env.connectUnity()
        print("Unity Connected ...")
    
    # DEBUG
    # Define expert policy
    if args.train:
        n_steps_temp = 250 # 250
        expert_first = PPO(
            tensorboard_log=log_dir,
            policy=MultiInputPolicy,
            policy_kwargs=dict(
                activation_fn=torch.nn.ReLU,
                net_arch=dict(pi=[256, 256], vf=[512, 512]),
                log_std_init=-0.5,
                use_expln=True, # to solve 'nan' cases 
            ),
            env=train_env,
            learning_rate= linear_schedule(7e-4), #lr_schedule, #3e-4,
            gae_lambda=0.95,
            gamma=0.99,
            n_steps=n_steps_temp,#250,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            batch_size=n_steps_temp * cfg["simulation"]["num_envs"],
            clip_range=0.2,
            use_sde=False,  # don't use (gSDE), doesn't work
            verbose=1,
            device="cuda",
        )
        
        # Debug Messages
        print(expert_first.policy)
        print(" Expert Learning ...")
        start_time = time.time()
        # Learn Expert Policy
        expert_first.learn(total_timesteps=int(20 * 1e8), log_interval=250)

        # Saved trained expert policy
        expert_first.save("ppo_expert")
        print("Expert policy saved ...")

        if args.render:
            train_env.disconnectUnity()
            print("Unity Disconnected ...")
            train_proc.terminate()
            print("Train Unity Process Terminated ...")
        
        end_time = time.time()
        print("Train takes : ", end_time - start_time)
        print("------------- TRAIN DONE ---------------")


    if args.teach:
        # --------------------- BC TRAINING -----------------------
         # expert = PPO.load("ppo_expert", env=train_env, device="cuda")
        weight = rsg_root + "/../saved/PPO_{0}/Policy/iter_{1:05d}.pth".format(args.trial, args.iter)
        env_rms = rsg_root +"/../saved/PPO_{0}/RMS/iter_{1:05d}.npz".format(args.trial, args.iter)
        saved_variables = torch.load(weight, map_location="cuda")
        
        # Create policy object
        # expert = MlpPolicy(**saved_variables["data"])
        print(saved_variables["data"])
        expert = MultiInputPolicy(**saved_variables["data"])
        
        # expert.action_net = torch.nn.Sequential(expert.action_net, torch.nn.Tanh())
        # Load weights
        expert.load_state_dict(saved_variables["state_dict"], strict=False)
        expert.to("cuda")
        # 
        # eval_env.load_rms(env_rms)

        # --------------------- BC TRAINING -----------------------
        expert_reward, _ = evaluate_policy(expert, train_env, n_eval_episodes=10)
        print(f"Expert Policy Reward: {expert_reward}")
        print(f"Expert Policy Reward: {expert_reward}")
        print(f"Expert Policy Reward: {expert_reward}")
        if expert_reward < 0:
            print("Expert Policy Reward is too low. BC training will not be performed.")
            #exit()

        # 3- Rollout
        # rollout.py icerisinde obs verisi okunurken ilk 65'ini alacak sekilde kesiyoruz.
        # Bu kodlar sadece bc sirasinda kullanildigi icin diger taraflari etkilemeyecektir
        rng = np.random.default_rng()
        rollouts = rollout.rollout(
            expert,
            train_env,
            rollout.make_sample_until(min_timesteps=None, min_episodes=50),
            rng=rng,
            unwrap=False,
        )

        transitions = rollout.flatten_trajectories(rollouts)

        # 4- Policy Initialization  
        # MultiInputLstmPolicy
        student_policy = MultiInputPolicy(
                observation_space=eval_env.observation_space,
                action_space=eval_env.action_space,
                # Set lr_schedule to max value to force error if policy.optimizer
                # is used by mistake (should use self.optimizer instead).
                lr_schedule=lambda _: th.finfo(th.float32).max,
                # features_extractor_class=extractor,
            )

        bc_trainer = bc.BC(
            policy=student_policy,
            #observation_space=train_env.observation_space["state"],  # Neden bunu bu sekilde yapmistik ?
            observation_space=eval_env.observation_space,
            action_space=eval_env.action_space,
            demonstrations=transitions,
            rng=rng,
            device="cuda",
        )

        # Train the Behavioral Clonning Model
        reward_before_training, _ = evaluate_policy(bc_trainer.policy, eval_env, 10)        
        bc_trainer.train(n_epochs=10)
        reward_after_training, _ = evaluate_policy(bc_trainer.policy, eval_env, 10)
        print(f"Reward before training: {reward_before_training}")
        print(f"Reward after training: {reward_after_training}")
        print(f"Expert Policy Reward: {expert_reward}")
        
        # Save the trained policy
        save_mode_custom(name="policy_imitation", policy=bc_trainer.policy)
        
    
    # args.imitation = 0
    if args.test:
        
        print("Test start ....")
        
        # Define device
        device = get_device("auto")
        
        # Open Unity
        if args.render:
            test_proc = subprocess.Popen(os.environ["FLIGHTMARE_PATH"] + "/flightrender/RPG_Flightmare.x86_64")
        
        if not args.imitation:
            # SB3 Policy Path
            weight = rsg_root + "/../saved/PPO_{0}/Policy/iter_{1:05d}.pth".format(args.trial, args.iter)
            env_rms = rsg_root +"/../saved/PPO_{0}/RMS/iter_{1:05d}.npz".format(args.trial, args.iter)
        else:
            # Imitation Policy Path
            weight = "/home/gazi13/catkin_ws_agile/src/agile_flight/envtest/save_imitation/policy_imitation/policy_imitation.pth"
            env_rms = None
        
        # # 1- Load Policy from pth file
        saved_variables = torch.load(weight, map_location=device)
        demo_policy = MultiInputPolicy(**saved_variables["data"])
        # @TODO: Why tanh is used? 
        # demo_policy.action_net = torch.nn.Sequential(demo_policy.action_net, torch.nn.Tanh())
        
        # Load weights
        demo_policy.load_state_dict(saved_variables["state_dict"], strict=False)
        demo_policy.to(device)
        if env_rms is not None:
            eval_env.load_rms(env_rms)
        
        # 2- Load Policy from zip file 
        # demo_policy = PPO.load("bc_policy_1", env=eval_env, device="cuda")

        # 3- Load Custom Model - Imitation
        # from imitation.policies.serialize import load_policy, policy_registry
        # from stable_baselines3.common import policies
        # def my_policy_loader(venv, some_param: int) -> policies.BasePolicy:
        #     # load your policy here
        #     return policy
        # policy_registry.register("my-policy", my_policy_loader)
        # demo_policy = load_policy("ppo", eval_env, path="ppo_expert.zip")
        



        # Evaluate Policy
        if not args.render or True:
            demo_policy_reward, _ = evaluate_policy(demo_policy, eval_env, 10)
            print(weight)
            print(f"Test Policy Reward: {demo_policy_reward}")
            print(f"Test Policy Reward: {demo_policy_reward}")
            print(f"Test Policy Reward: {demo_policy_reward}")

        # Test Policy
        if not args.train:
            # @TODO: LSTM ILE TEST ICIN - model.predict DUZENLENMELI !!!
            # @TODO: LSTM ILE TEST ICIN - model.predict DUZENLENMELI !!!
            # @TODO: LSTM ILE TEST ICIN - model.predict DUZENLENMELI !!!
            test_policy(eval_env, demo_policy, render=args.render)

        if args.render:
            test_proc.terminate()

        print("Test DONE ....")

if __name__ == "__main__":
    main()

