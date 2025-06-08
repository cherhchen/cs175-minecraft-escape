import gym
import numpy as np
from pathlib import Path
import os
import ray
import ray.tune as tune
import argparse
from env import Env
from ray.tune import SyncConfig
from ray.rllib.agents.callbacks import DefaultCallbacks
import tensorflow as tf


def create_env(config):
    xmls = config["mission_xml"]
    env = Env(xmls=xmls, num_workers=config["num_workers"], ms_per_tick=50)
    return env

def stop_check(trial, result):
    return result["episode_reward_mean"] >= 250

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mission_path", type=str)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument('--checkpoint_file', required=False, type=str)
    args = parser.parse_args()

    tune.register_env("escape_room_v7", create_env)


    ray.init(log_to_driver=False,
             include_dashboard=False)

    tune.run(
        run_or_experiment="DQN",
        config={
            "log_level": "WARN",
            "env": "escape_room_v7",
            "env_config": {
                "mission_xml": args.mission_path,
                "num_workers": args.num_workers},
            "framework": "tf",
            "num_gpus": 1,
            "num_workers": args.num_workers,
            "num_cpus_per_worker": 1,
            "tf_session_args": {
                "gpu_options": {
                    "allow_growth": True
                }
            },
            "double_q": True,
            "dueling": True,
            "explore": True,
            "prioritized_replay": True,
            "prioritized_replay_alpha": 0.3,
            "prioritized_replay_beta": 0.8,
            "prioritized_replay_eps": 1e-6,
            "buffer_size": 50000,
            "learning_starts": 5000,  # allow exploration first
            "train_batch_size": 64,
            "exploration_config": {
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.1,
                "epsilon_timesteps": 300000
            }
        },
        stop=stop_check,
        checkpoint_freq=1,
        checkpoint_at_end=True,
        resume=True,
        # restore=args.checkpoint_file,
        local_dir='./logs',
        keep_checkpoints_num=5,
        sync_config=SyncConfig(sync_to_driver=False))
    
    print("Finished Training")
    ray.shutdown()