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
import torch as th

from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent import MlpLstmPolicy, MultiInputLstmPolicy

def configure_random_seed(seed, env=None):
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

def main():
    args = parser().parse_args()

    # Save the configuration and other files
    rsg_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = rsg_root + "/../saved"
    os.makedirs(log_dir, exist_ok=True)
    w_path = rsg_root + "/../saved/RecurrentPPO_{0}/Policy/iter_{1:05d}.pth".format(args.trial, args.iter)
    
    if not os.path.exists(w_path):
        print(" TRAIN MODEL ")
        args.train = 1
        args.teach = 0
        args.test = 1
        args.render = 0
    else:
        print(" TEST MODEL ")
        args.train = 0
        args.teach = 0
        args.test = 1
        args.render = 1
        
    # Load configurations
    cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] + "/flightpy/configs/vision/config.yaml", "r"))
    
    # Change config according to args
    if not args.train and not args.teach:
        cfg["simulation"]["num_envs"] = 1
    else:
        cfg["simulation"]["num_envs"] = 300 #100
        cfg["simulation"]["num_threads"] = 50
    if args.render:
        cfg["unity"]["render"] = "yes"

    # DEBUG - PERFORMANS
    # cfg["simulation"]["num_threads"] = 10
    # cfg["simulation"]["sim_dt"] = 0.03
    # cfg["simulation"]["max_t"] = 20.0
    cfg["rgb_camera"]["width"] = 800 # 160
    cfg["rgb_camera"]["height"] = 600 # 80

    # Training environment
    train_env = AgileEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    train_env = wrapper.FlightEnvVec(train_env)
    print("Train environment defined ...")

    # Set random seed
    configure_random_seed(args.seed, env=train_env)
    
    # Evaluation environment
    old_num_envs = cfg["simulation"]["num_envs"]
    cfg["simulation"]["num_envs"] = 1
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
    
    # Define expert policy
    if args.train:
        n_steps_temp = 250 # 250
        expert_first = RecurrentPPO(
            tensorboard_log=log_dir,
            policy=MultiInputLstmPolicy,
            policy_kwargs=dict(
                activation_fn=torch.nn.ReLU,
                net_arch=dict(pi=[256, 256], vf=[512, 512]),
                # net_arch=[dict(pi=[64, 64], vf=[64, 64])],
                log_std_init=-0.5,
                use_expln=True, # BEN EKLEDIM
            ),
            env=train_env,
            # eval_env=eval_env,
            # use_tanh_act=True,
            gae_lambda=0.95,
            gamma=0.99,
            n_steps=n_steps_temp,#250,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            batch_size=n_steps_temp * cfg["simulation"]["num_envs"],#25000,
            clip_range=0.2,
            use_sde=False,  # don't use (gSDE), doesn't work
            # env_cfg=cfg,
            verbose=1,
            # seed=1,
            device="cuda",
        )
        
        # Debug Messages
        print(expert_first.policy)
        print(" Expert Learning ...")
        start_time = time.time()
        # Learn Expert Policy
        expert_first.learn(total_timesteps=int(20 * 1e8), log_interval=100)

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
        # # python3 -m run_imitation_bc --render 0 --train 0 --trial 61 --iter 1700 
        # # python3 -m run_imitation_bc --render 1 --train 0 --trial 61 --iter 1700 
        # python3 -m run_imitation_bc --render 0 --train 0 --teach 1 --trial 5 --iter 2000

        # --------------------- BC TRAINING -----------------------

         # expert = PPO.load("ppo_expert", env=train_env, device="cuda")
        weight = rsg_root + "/../saved/PPO_{0}/Policy/iter_{1:05d}.pth".format(args.trial, args.iter)
        env_rms = rsg_root +"/../saved/PPO_{0}/RMS/iter_{1:05d}.npz".format(args.trial, args.iter)
        saved_variables = torch.load(weight, map_location="cuda")
        
        # Create policy object
        # expert = MlpPolicy(**saved_variables["data"])
        print(saved_variables["data"])
        expert = MultiInputLstmPolicy(**saved_variables["data"])
        
        #
        # expert.action_net = torch.nn.Sequential(expert.action_net, torch.nn.Tanh())
        # Load weights
        expert.load_state_dict(saved_variables["state_dict"], strict=False)
        expert.to("cuda")
        # 
        eval_env.load_rms(env_rms)

        # --------------------- BC TRAINING -----------------------
        expert_reward, _ = evaluate_policy(expert, eval_env, n_eval_episodes=10)
        print(f"Expert Policy Reward: {expert_reward}")
        print(f"Expert Policy Reward: {expert_reward}")
        print(f"Expert Policy Reward: {expert_reward}")
        if expert_reward < 0:
            print("Expert Policy Reward is too low. BC training will not be performed.")
            #exit()
        
        print("train_env.observation_space['state'].shape:", train_env.observation_space["state"].shape)
        print("train_env.observation_space['state'].shape:", train_env.observation_space["state"])  
        # 3- Rollout
        rng = np.random.default_rng()
        rollouts = rollout.rollout(
            expert,
            train_env,
            rollout.make_sample_until(min_timesteps=None, min_episodes=50),
            rng=rng,
            unwrap=False,
        )
        print("Rollout Done ...")

        transitions = rollout.flatten_trajectories(rollouts)
        print("Transitions Done ...")

        # 4- bc_trainer.policy'nin  Tanimlanmasi Gerekiyor 
        # Policy default olarak bu sekilde tanimlaniyor 
        # SB3 Custom Policy Tanimi kullanilabilir mi
        """ 
        extractor = (
                torch_layers.CombinedExtractor
                if isinstance(observation_space, gym.spaces.Dict)
                else torch_layers.FlattenExtractor
            )
            student_policy = policy_base.FeedForward32Policy(
                observation_space=observation_space,
                action_space=action_space,
                # Set lr_schedule to max value to force error if policy.optimizer
                # is used by mistake (should use self.optimizer instead).
                lr_schedule=lambda _: th.finfo(th.float32).max,
                features_extractor_class=extractor,
            )
        """

        # train_env.observation_space "spaces.Space" tipinde tanimli olmali
        # mantik olarak MultiInputPolicy hem goruntu hem deger alabildigi icin
        # observation_space sabit bir tip olmamali


        student_policy = MultiInputLstmPolicy(
                observation_space=train_env.observation_space,
                action_space=train_env.action_space,
                # Set lr_schedule to max value to force error if policy.optimizer
                # is used by mistake (should use self.optimizer instead).
                lr_schedule=lambda _: th.finfo(th.float32).max,
                # features_extractor_class=extractor,
            )
        print("Student Policy Created ...")

        bc_trainer = bc.BC(
            policy=student_policy,
            #observation_space=train_env.observation_space["state"],  # Neden bunu bu sekilde yapmistik ?
            observation_space=train_env.observation_space,
            action_space=train_env.action_space,
            demonstrations=transitions,
            rng=rng,
            device="cuda",
        )
        print("BC Trainer Created ...")

        print("evaluate_policy start ... ")
        reward_before_training, _ = evaluate_policy(bc_trainer.policy, eval_env, 10)
        print(f"Reward before training: {reward_before_training}")
        
        # Train the Behavioral Clonning Model
        print("Training start ... ")
        bc_trainer.train(n_epochs=10)
        print("Training Done ... ")
        
        reward_after_training, _ = evaluate_policy(bc_trainer.policy, eval_env, 10)
        print(f"Reward before training: {reward_before_training}")
        print(f"Reward after training: {reward_after_training}")
        print(f"Expert Policy Reward: {expert_reward}")
        
        
        # Save the trained policy
        save_mode_custom(name="policy_imitation", policy=bc_trainer.policy)


        # bc_trainer.policy.save("bc_policy")
        # from stable_baselines3.common.save_util import save_to_zip_file
        # from imitation.util.util import save_policy
        # save_policy(bc_trainer.policy, "bc_policy_1")
        # # save_stable_model(bc_trainer.policy, "bc_policy_1")
        # # data = bc_trainer.policy._get_constructor_parameters()
        # # state_dict = bc_trainer.policy.state_dict()
        # # save_to_zip_file("bc_policy_1", data=data, params=state_dict)
        
    if args.test:
        
        print("Test start ....")
        
        # Define device
        device = get_device("auto")
        
        # Open Unity
        if args.render:
            test_proc = subprocess.Popen(os.environ["FLIGHTMARE_PATH"] + "/flightrender/RPG_Flightmare.x86_64")

        # SB3 Policy Path
        weight = rsg_root + "/../saved/RecurrentPPO_{0}/Policy/iter_{1:05d}.pth".format(args.trial, args.iter)
        env_rms = rsg_root +"/../saved/RecurrentPPO_{0}/RMS/iter_{1:05d}.npz".format(args.trial, args.iter)
        print(weight)
        print(env_rms)
        # Imitation Policy Path
        # weight = "/home/gazi13/catkin_ws_agile/src/agile_flight/envtest/save_imitation/policy_imitation/policy_imitation.pth"
        # env_rms = None
        
        # # 1- Load Policy from pth file
        saved_variables = torch.load(weight, map_location=device)
        demo_policy = MultiInputLstmPolicy(**saved_variables["data"])
        
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
            print(f"Test Policy Reward: {demo_policy_reward}")
            print(f"Test Policy Reward: {demo_policy_reward}")
            print(f"Test Policy Reward: {demo_policy_reward}")
            print(weight)
            print(env_rms)

        # Test Policy
        if not args.train:
            test_policy(eval_env, demo_policy, render=args.render)

        if args.render:
            test_proc.terminate()

        print("Test DONE ....")

if __name__ == "__main__":
    main()
