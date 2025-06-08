import gym
import MalmoPython
import random
import time
import numpy as np
from tensorboardX import SummaryWriter
from gym.spaces import Box
import json
import os
from pathlib import Path

PORT = 10000

class ActionSpace(gym.spaces.Discrete):
    def __init__(self):
        self.actions = ["move 1", "jumpmove", "turn 1", "turn -1"]
        gym.spaces.Discrete.__init__(self, len(self.actions))
    
    def sample(self):
        return random.randint(0, len(self.actions) - 1)
    
    def __getitem__(self, action):
        return self.actions[action]
    
    def __len__(self):
        return len(self.actions)
    
class Env(gym.Env):
    def __init__(self, xmls, num_workers, ms_per_tick = 50):
        super(Env, self).__init__()
        if "xml" in xmls:
            self.xmls = xmls
        else:
            self.xmls = [
                os.path.join(xmls, f)
                for f in os.listdir(xmls)
                if f.endswith(".xml")
            ]
        self.ms_per_tick = ms_per_tick
        self.room_shape = (9, 7, 6)
        self.block_to_idx = {"air": 0, "purpur_block": 1, "stone_slab": 2, "water": 3, "gold_block": 4}
        self.num_block_types = len(self.block_to_idx)
        obs_space_size = (self.num_block_types + 1) * np.prod(self.room_shape) + 5
        self.observation_space = Box(
            low=-1,
            high=1,
            shape=(obs_space_size,),
            dtype=np.float32
        )
        self.action_space = ActionSpace()
        self.agent_host = MalmoPython.AgentHost()
        self.mission_record = MalmoPython.MissionRecordSpec()
        self.mission_record.recordRewards()
        self.mission_record.recordObservations()

        self.pool = MalmoPython.ClientPool()
        for i in range(num_workers):
            client = MalmoPython.ClientInfo("127.0.0.1", PORT + i)
            self.pool.add(client)
        
        self.writer = SummaryWriter(log_dir="./logs/all_missions")
        self.mission_id = 0
        self.step_count = 0
        self.action_repeat = 10
        self.prev_y = None
        self.max_y = None

    def reset(self):
        self.mission_id += 1
        self.step_count = 0
        self.prev_y = None
        self.max_y = None
        if type(self.xmls) is list:
            xml = random.choices(self.xmls, weights = [0.30, 0.70])[0]
        else:
            xml = self.xmls
        mission_xml = Path(xml).read_text()
        mission = MalmoPython.MissionSpec(mission_xml, True)
        max_retries = 10
        for retry in range(max_retries):
            try:
                self.agent_host.startMission(mission, self.pool, self.mission_record, 0, "escape_room")
                break
            except:
                if retry == max_retries - 1:
                    print("Error starting mission")
                    exit(1)
                else:
                    time.sleep(2)

        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(self.ms_per_tick / 1000)
            world_state = self.agent_host.getWorldState()
        
        grid, _, _, agent_pos = self._process_state(False)
        self.last_obs = self._get_obs(grid, agent_pos)

        return self.last_obs
    
    def render(self, mode=None):
        pass

    def step(self, action_idx):
        action = self.action_space[action_idx]
        if action == "jumpmove":
            self.agent_host.sendCommand("move 1")
            time.sleep(0.3)
            self.agent_host.sendCommand("jump 1")
            time.sleep(0.2)
            self.agent_host.sendCommand("jump 0")
            self.agent_host.sendCommand("move 0")
        else:
            self.agent_host.sendCommand(action)
            time.sleep(0.2)
            self.agent_host.sendCommand("turn 0")
            self.agent_host.sendCommand("move 0")
        
        grid, reward, done, agent_pos = self._process_state()
        if agent_pos is not None:
            _, y_pos, _, _ = agent_pos
        else:
            y_pos = None

        if self.prev_y is not None and y_pos is not None:
            if y_pos > self.prev_y and (self.max_y is None or y_pos > self.max_y):
                reward += 25
                self.max_y = y_pos
            elif y_pos < self.prev_y:
                reward -= 15
        self.prev_y = y_pos
        
        self.last_obs = self._get_obs(grid, agent_pos)
        
        self.writer.add_text(f"Mission/{self.mission_id}/Action", action, self.step_count)
        self.writer.add_scalar(f"Mission/{self.mission_id}/Reward", reward, self.step_count)

        self.step_count += 1

        return self.last_obs, reward, done, {}
        
    def close(self):
        self.agent_host.sendCommand("quit")
        _,_,_,_ = self._process_state()
        self.writer.close()

    def _process_state(self, get_reward=True):
        reward = 0
        attempts = 0
        post_done = False
        grid = None
        done = False
        rewardFlag = False
        gridFlag = False
        agent_pos = None
        while attempts < 100:
            time.sleep(self.ms_per_tick / 1000)
            world_state = self.agent_host.getWorldState()
            if world_state.number_of_observations_since_last_state > 0:
                try:
                    obs_json = world_state.observations[-1].text
                    data = json.loads(obs_json)
                    grid = data["fullRoom"]
                    agent_pos = (int(round(data["XPos"])), int(round(data["YPos"])), int(round(data["ZPos"])), data["Yaw"])
                    gridFlag = True
                except:
                    pass
            if world_state.number_of_rewards_since_last_state > 0:
                reward += world_state.rewards[-1].getValue()
                rewardFlag = True
            done = done or not world_state.is_mission_running

            if done:
                if not get_reward:
                    break
                if post_done:
                    break
                post_done = True
                continue
            else:
                if get_reward and rewardFlag and gridFlag:
                    break
                elif (not get_reward) and gridFlag:
                    break
            attempts += 1
        return grid, reward, done, agent_pos
    
    def _get_obs(self, raw_grid, agent_pos):
        if raw_grid is None or len(raw_grid) != np.prod(self.room_shape):
            raw_grid = ["air"] * np.prod(self.room_shape)

        grid_idx = np.array(
            [self.block_to_idx.get(b, 0) for b in raw_grid],
            dtype=np.int32
        ).reshape(self.room_shape)

        onehot = np.eye(self.num_block_types, dtype=np.float32)[grid_idx]
        onehot = onehot.transpose(3,0,1,2)

        agent_channel = np.zeros_like(onehot[0], dtype=np.float32)  # shape: (H, W, D)
        yaw_feat = np.zeros(2, dtype=np.float32)
        agent_subblock = np.zeros(3, dtype=np.float32)
        if agent_pos is not None:
            x, y, z, yaw = agent_pos
            x_idx = np.clip(int(x), 0, self.room_shape[1] - 1)  # X axis
            y_idx = np.clip(int(y), 0, self.room_shape[0] - 1)  # Y axis
            z_idx = np.clip(int(z), 0, self.room_shape[2] - 1)  

            agent_channel[y_idx, x_idx, z_idx] = 1
            
            agent_subblock = np.array([
            x / (self.room_shape[1] - 1),   # normalize x: 4.73 / 6
            y / (self.room_shape[0] - 1),   # normalize y: 2.00 / 8
            z / (self.room_shape[2] - 1)   # normalize z: 3.10 / 5
            ], dtype=np.float32)

            rad = np.deg2rad(yaw)
            yaw_feat = np.array([np.sin(rad), np.cos(rad)], dtype=np.float32)


        obs = np.concatenate([onehot, agent_channel[None, ...]], axis=0)
        obs_flat = obs.flatten().astype(np.float32)
        final_obs = np.concatenate([obs_flat, agent_subblock, yaw_feat], axis=0).astype(np.float32)

        return final_obs
