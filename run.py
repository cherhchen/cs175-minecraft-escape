import ray
import ray.tune as tune
from ray.rllib import rollout
from ray.tune.registry import get_trainable_cls
import gym

import argparse
from pathlib import Path
import random
import time
from env import Env
from ray.tune import SyncConfig

def create_env(config):
    xml = config["mission_xml"]
    env = Env(xmls=xml, num_workers = config["num_workers"], ms_per_tick=50)
    return env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mission_path", type=str)
    parser.add_argument('--checkpoint_file', type=str)
    args = parser.parse_args()

    ray.init(log_to_driver=False,
             include_dashboard=False)

    tune.register_env("escape_room2", create_env)
    cls = get_trainable_cls("DQN")
    agent = cls(env="escape_room2", config={
        "env_config": {
            "mission_xml": args.mission_path,
            "num_workers": 1},
        "framework": "tf",
        "num_gpus": 1,
        "num_workers": 0,
        "double_q": True,
        "dueling": True,
        "explore": False
    })
    agent.restore(args.checkpoint_file)
    env = agent.workers.local_worker().env
    obs = env.reset()
    done = False
    total_reward = 0
    start = time.time()
    while not done:
        action = agent.compute_single_action(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
    end = time.time()
    duration = end - start
    env.close()
    agent.stop()
    print("Total Reward:", total_reward)
    print("Time elapsed:", duration, "seconds")

    ray.shutdown()