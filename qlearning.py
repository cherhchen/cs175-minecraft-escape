import MalmoPython
import time
import random
import math
import sys
import json
from collections import defaultdict


def create_mission():
    mission_xml = '''<Mission xmlns="http://ProjectMalmo.microsoft.com"
             xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

      <About>
        <Summary>Escape room by placing Redstone torch</Summary>
      </About>

      <ServerSection>
        <ServerInitialConditions>
          <Time>
            <StartTime>1000</StartTime>
            <AllowPassageOfTime>false</AllowPassageOfTime>
          </Time>
        </ServerInitialConditions>

        <ServerHandlers>
          <FlatWorldGenerator generatorString="3;2,1,41;23;"/>
          <DrawingDecorator>          
            <!--  Draw Room  -->
            <DrawCuboid x1="0" y1="1" z1="0" x2="6" y2="6" z2="5" type="stone"/>
            <DrawCuboid x1="1" y1="2" z1="1" x2="5" y2="6" z2="4" type="air"/>
            
            <!-- Add water to the floor  -->
            <DrawCuboid x1="1" y1="1" z1="1" x2="5" y2="2" z2="4" type="water"/>
              
            <!--  Add starting platform   -->
            <DrawBlock x="3" y="3" z="3" type="purpur_slab"/>
            
            <!--  Draw Parkour Platforms -->
            <DrawBlock x="3" y="2" z="3" type="stone_slab"/>
            <DrawBlock x="5" y="3" z="4" type="stone_slab"/>
            <DrawBlock x="4" y="3" z="4" type="stone_slab"/>
            <DrawBlock x="2" y="4" z="4" type="stone_slab"/>
            <DrawBlock x="1" y="5" z="3" type="stone_slab"/>
            <DrawBlock x="1" y="6" z="1" type="stone_slab"/>
            
            
            <!--  Goal  -->
            <DrawBlock x="1" y="6" z="0" type="gold_block"/>
            
            
            <!-- Carve space in room for door -->
            <!-- <DrawBlock x="1" y="7" z="0" type="air"/> -->
            <!-- <DrawBlock x="1" y="8" z="0" type="air"/> -->

          </DrawingDecorator>

          <ServerQuitWhenAnyAgentFinishes/>
        </ServerHandlers>
      </ServerSection>

      <AgentSection mode="Survival">
        <Name>EscapeAgent</Name>
        <AgentStart>
          <Placement x="3" y="4" z="3" pitch="0" yaw="0"/>
        </AgentStart>

        <AgentHandlers>
          <ContinuousMovementCommands/>
          <InventoryCommands/>
          <ObservationFromFullStats/>
          <ObservationFromGrid>
            <Grid name="under_agent">
              <min x="0" y="-1" z="0"/>
              <max x="0" y="-1" z="0"/>
            </Grid>
          </ObservationFromGrid>
          
          <!-- <RewardForTouchingBlockType>
            <Block type="gold_block" reward="100.0"/>
            <Block type="stone_slab" reward="0"/>
            <Block type="flowing_water" reward="0"/>
            <Block type="stone" reward="0"/>
          </RewardForTouchingBlockType> -->
          <MissionQuitCommands/>
          <AgentQuitFromTouchingBlockType>
            <Block type="gold_block" description="Goal_found"/>
          </AgentQuitFromTouchingBlockType>
        </AgentHandlers>
      </AgentSection>
    </Mission>'''
    return mission_xml

class Prisoner(object):
    def __init__(self, alpha=0.3, gamma=0.99, n=1):
        """Constructing an RL agent.

        Args
            alpha:  <float>  learning rate      (default = 0.3)
            gamma:  <float>  value decay rate   (default = 1)
            n:      <int>    number of back steps to update (default = 1)
        """
        self.epsilon = 0.3  # chance of taking a random action instead of the best
        self.q_table = {}
        self.n, self.alpha, self.gamma = n, alpha, gamma
        self.inventory = defaultdict(lambda: 0, {})
        
        # Define actions
        # self.actions = {0: ("move 1",), 1: ("jump 1",), 2: ("turn 1",), 3: ("turn -1",)}
        self.actions = ["move 1", "jump 1", "turn 1", "turn -1"]
        self.num_actions = len(self.actions)

   
    def get_state(self, obs):
        return (
            int(round(obs.get("XPos", 0))),
            int(round(obs.get("YPos", 0))),
            int(round(obs.get("ZPos", 0))),
            int(round(obs.get("Yaw", 0) / 90.0)) % 4  # 0 = North, 1 = East, 2 = South, 3 = West
        )
    
    
    # Action selection
    def choose_action(self, state):
        # print(self.q_table, "\n")
        if random.random() < self.epsilon or state not in self.q_table:
            # i = random.randint(0, self.num_actions - 1)
            # return i
            return random.choice(self.actions)
        return max(self.q_table[state], key=self.q_table[state].get)
    
    
    # Q-value update
    def update_q(self, state, action, reward, next_state):
        print("state:", state, "reward:", reward, "action:", action, "next_state:", next_state)
        
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0 for a in self.actions}
            
        # Add heuristic to adjust reward
        heuristic_reward = self.calc_heuristic(next_state)
        w = 0.1
        adjusted_reward = reward + w * heuristic_reward
        
        max_future_q = max(self.q_table[next_state].values())
        self.q_table[state][action] += self.alpha * (adjusted_reward + self.gamma * max_future_q - self.q_table[state][action])
        
    def calc_reward(self, obs, prev_y):
        block_below = obs.get("under_agent", [""])[0]
        print(block_below)
        y = int(obs.get("YPos", 0))
        
        reward = 0
        
        if block_below == "water" or block_below == "grass":
            reward += -100
        elif block_below == "stone_slab":
            reward += 50
        elif block_below == "purpur_slab":
            reward += 10
            
        if y > prev_y:
            reward += 25
        elif y < prev_y:
            reward -= 25
        
        return reward
    
    def calc_heuristic(self, state):
        # Gold block is at (1, 6, 0)
        goal_x, goal_y, goal_z = 1, 6, 0
        x, y, z, _ = state
        # Calculate manhattan distance (measures distance between two points in a grid-like space)
        manhattan_dist = abs(goal_x - x) + abs(goal_y - y) + abs(goal_z - z)
        # Smaller distance = higher heuristic
        return -1 * manhattan_dist
        
        
#     def is_on_start_block(self, obs):
#         x = int(obs.get("XPos", 0))
#         y = int(obs.get("YPos", 0))
#         z = int(obs.get("ZPos", 0))
#         return x == 3 and y == 2 and z == 3
        
#     def is_in_water(self, obs):
#         x = int(obs.get("XPos", 0))
#         y = int(obs.get("YPos", 0))
#         z = int(obs.get("ZPos", 0))
#         return not self.is_on_start_block(obs) and y == 2 and 1 <= x <= 5 and 1 <= z <= 4
    
    
if __name__ == '__main__':
    # Launch Malmo and run episodes
    random.seed(0)
    my_client_pool = MalmoPython.ClientPool()
    my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10000))
    
    agent_host = MalmoPython.AgentHost()
    
    try:
        agent_host.parse(sys.argv)
    except RuntimeError as e:
        print('ERROR:', e)
        print(agent_host.getUsage())
        exit(1)
    if agent_host.receivedArgument("help"):
        print(agent_host.getUsage())
        exit(0)

    episodes = 5
    prisoner = Prisoner()
    for episode in range(episodes):
        print("Episode" + str(episode+1))

        mission_xml = create_mission()
        mission_spec = MalmoPython.MissionSpec(mission_xml, True)
        mission_record = MalmoPython.MissionRecordSpec()

        # Attempt to start mission
        for retry in range(3):
            try:
                agent_host.startMission(mission_spec, my_client_pool, mission_record, 0, "Prisoner")
                break
            except RuntimeError as e:
                if retry == 2:
                    print("Error starting mission", e)
                    print("Is the game running?")
                else:
                    time.sleep(2)

        # Wait for mission to start
        print("Waiting for mission to start...")
        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = agent_host.getWorldState()

        total_reward = 0
        prev_state = None
        prev_action = None
        prev_y = None
        
        local_time = time.localtime()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time, "(UTC)")
        print("Start Time:", formatted_time)
        start = time.time()
        
        # Main loop
        while world_state.is_mission_running:
            world_state = agent_host.getWorldState()
            if world_state.number_of_observations_since_last_state > 0:
                obs_text = world_state.observations[-1].text
                obs = json.loads(obs_text)
                state = prisoner.get_state(obs)
                
                if prev_y is None:
                    prev_y = state[1]

                action = prisoner.choose_action(state)
                agent_host.sendCommand(action)
                # for command in prisoner.actions[action]:
                    # print("command: ", command)
                    # agent_host.sendCommand(command)
                    # time.sleep(0.4)
                time.sleep(0.2)

                # reward = sum(r.getValue() for r in world_state.rewards)
                # total_reward += reward
                reward = prisoner.calc_reward(obs, prev_y)
                
                if prev_state != None and prev_action != None:
                    prisoner.update_q(prev_state, prev_action, reward, state)
                
                prev_state = state
                prev_action = action
                
        end = time.time()
        length = end - start
        local_time = time.localtime()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time, "(UTC)")
        print("End Time:", formatted_time)
        print("Time elapsed:", length, "seconds")
        print("Goal reached!")
        print("Final Q-table:", prisoner.q_table)

print("Training complete!")
