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
        <Summary>Parkour Escape Room 2</Summary>
      </About>

      <ServerSection>
        <ServerInitialConditions>
          <Time>
            <StartTime>1000</StartTime>
            <AllowPassageOfTime>false</AllowPassageOfTime>
          </Time>
        </ServerInitialConditions>

        <ServerHandlers>
          <FlatWorldGenerator generatorString="3;2,1;24;"/>
          <DrawingDecorator>
            <!--  Draw Room  -->
            <DrawCuboid x1="0" y1="1" z1="0" x2="6" y2="10" z2="5" type="stone"/>
            <DrawCuboid x1="1" y1="2" z1="1" x2="5" y2="10" z2="4" type="air"/>
            
            <!-- Add water to the floor  -->
            <DrawCuboid x1="1" y1="2" z1="1" x2="5" y2="2" z2="4" type="water"/>
              
            <!--  Add starting platform   -->
            <DrawBlock x="4" y="3" z="3" type="purpur_slab"/>
            
            <!--  Draw Parkour Platforms -->           
            <DrawBlock x="5" y="4" z="4" type="stone_slab"/>
            <DrawBlock x="4" y="4" z="4" type="stone_slab"/>
            <DrawBlock x="2" y="5" z="4" type="stone_slab"/>
            <DrawBlock x="1" y="6" z="3" type="stone_slab"/>
            <DrawBlock x="1" y="7" z="1" type="stone_slab"/>
            
            <!--  Goal  -->
            <DrawBlock x="1" y="7" z="0" type="gold_block"/>
              
            <!-- Carve space in room for door -->
            <DrawBlock x="1" y="8" z="0" type="air"/>
            <DrawBlock x="1" y="9" z="0" type="air"/>

          </DrawingDecorator>

          <ServerQuitWhenAnyAgentFinishes/>
        </ServerHandlers>
      </ServerSection>

      <AgentSection mode="Survival">
        <Name>EscapeAgent</Name>
        <AgentStart>
            <!-- Place agent on top of starting platform -->
            <Placement x="4" y="4" z="3" pitch="0" yaw="0"/>
        </AgentStart>

        <AgentHandlers>
          <ContinuousMovementCommands/>
          <InventoryCommands/>
          <MissionQuitCommands/>
          <ObservationFromFullStats/>
            
          <!-- check if the agent is inside a water block (they fell) -->
          <ObservationFromGrid>
            <Grid name="current_block">
              <min x="0" y="0" z="0"/>
              <max x="0" y="0" z="0"/>
            </Grid>
            <Grid name="agent_nearby_blocks">
              <min x="-1" y="0" z="-1"/>
              <max x="1" y="2" z="1"/>
            </Grid>
          </ObservationFromGrid>
            
          <AgentQuitFromTouchingBlockType>
            <Block type="gold_block" description="Goal_found"/>
          </AgentQuitFromTouchingBlockType>
        </AgentHandlers>
      </AgentSection>
    </Mission>
'''
    return mission_xml

def reshape_to_3d(flat_list):
    grid = []
    i = 0
    for y in range(3):  # y = height
        layer = []
        for z in range(3):  # z = depth
            row = []
            for x in range(3):  # x = width
                row.append(flat_list[i])
                i += 1
            layer.append(row)
        grid.append(layer)
    return grid  # grid[y][z][x]

def reshape_nearby_blocks(nb):
    nearby_blocks = [tuple(nb[:9]), tuple(nb[9:18]), tuple(nb[18:])]
    return nearby_blocks

class Prisoner(object):
    def __init__(self, alpha=0.3, gamma=0.99, n=1):
        """Constructing an RL agent.

        Args
            alpha:  <float>  learning rate      (default = 0.3)
            gamma:  <float>  value decay rate   (default = 1)
            n:      <int>    number of back steps to update (default = 1)
        """
        self.epsilon = 0.3  # chance of taking a random action instead of the best
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.95 # implement epsilon decay (decay by 5% after each episode)
        self.q_table = {}
        self.n, self.alpha, self.gamma = n, alpha, gamma
        self.inventory = defaultdict(lambda: 0, {})
        
        self.user_escaped = False
        self.num_actions_taken = 0
        self.reward = 0
        self.TIMEOUT = 7200
        
        # Define actions
        # self.actions = {0: ("move 1",), 1: ("jump 1",), 2: ("turn 1",), 3: ("turn -1",)}
        self.actions = ["move 1", "move -1", "jump 1", "strafe -1", "strafe 1"]
#         self.actions = ["move 1", "move -1", "jump 1", "strafe -1", "turn 1"]
        self.num_actions = len(self.actions)

    def inc_num_actions_taken(self):
        self.num_actions_taken += 1
        
    def get_num_actions_taken(self):
        return self.num_actions_taken
   
    def get_state(self, obs):
        # Return agent's location and nearby blocks from a 3x3 cube around the agent
        nearby_blocks = tuple(reshape_nearby_blocks(obs['agent_nearby_blocks']))

        return (
            int(round(obs.get("XPos", 0))),
            int(round(obs.get("YPos", 0))),
            int(round(obs.get("ZPos", 0))),
            int(round(obs.get("Yaw", 0) / 90.0)) % 4,  # 0 = North, 1 = East, 2 = South, 3 = West
            nearby_blocks
        )
    
    def fell_in_water(self, obs):
        """Determine if the agent fell in water"""
        # check agent height and block below to see if they fell in water
        if obs["YPos"] < 3 and obs["current_block"][0] == "water":
        # Agent is not standing on standing platform, therefore, they are in water
            return True
        else:
            return False
      
    
   # Provide some guidance to random action based on nearby blocks
    def bias_random_action(self, nb):
        # nearby_blocks = 3x3
        # nearby_blocks[0] = floor
        # nearby_blocks[1] = agent level
        # nearby_blocks[2] = above agent's head
        
        # Useful when we do not restart mission after agent falls in the water
        if nb[1][3] == 'purpur_slab' or nb[1][6] == 'purpur_slab':
            # Turn right and jump when starting block is on the right in front of agent
            print("PURPUR RIGHT")
            return ('strafe 1', 'jump 1',)
        elif nb[1][5] == 'purpur_slab' or nb[1][8] == 'purpur_slab':
            # Turn left and jump when starting block is on the left in front of agent
            print("PURPUR LEFT")
            return ('strafe -1', 'jump 1',)
        elif nb[1][7] == 'purpur_slab':
            # Move forward and jump when starting block in front of agent (not sure if works or not)
            print("PURPUR FRONT")
            return ('move 1', 'move 1', 'jump 1',)
        
        # Useful when we do not restart mission after agent falls in the water
        if nb[1][1] == 'stone' and nb[0][4] == 'water':
            # Move forward if stone is behind agent
            print('STONE BACK')
            return 'move 1'
        if (nb[0][3] == 'stone' or nb[1][3] == 'stone') and nb[0][4] == 'water':
            # Move left if stone is right of agent
            print('STONE RIGHT')
            return 'strafe -1'
        if (nb[0][5] == 'stone' or nb[1][5] == 'stone') and nb[0][4] == 'water':
            # Move right if stone if left of agent
            print('STONE LEFT')
            return 'strafe 1'
        
        # Useful when agent is on the first slab stuck jumping at the left corner 
        if nb[2][5] == 'stone' and nb[2][8] == 'stone' and nb[2][7] == 'stone':
            print('STONE TOP RIGHT CORNER')
            return 'strafe 1'

        return None 
    
    # Action selection
    def choose_action(self, state):
        _, _, _, _, nearby_blocks = state
        if random.random() < self.epsilon or state not in self.q_table:
            biased_action = self.bias_random_action(nearby_blocks)
            # TODO: Maybe change probability of biased action?
            if biased_action and random.random() < 1:
                return biased_action
            return random.choice(self.actions)
        return max(self.q_table[state], key=self.q_table[state].get)
    
    
    # Q-value update
    def update_q(self, state, action, reward, next_state):
#         print("state:", state, "reward:", reward, "action:", action, "next_state:", next_state)
        print("reward:", reward, "action:", action)
        
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0 for a in self.actions}
            
        if action not in self.actions:
            self.q_table[state][action] = 0
        
        max_future_q = max(self.q_table[next_state].values())
        self.q_table[state][action] += self.alpha * (reward + self.gamma * max_future_q - self.q_table[state][action])

    def calc_reward(self, obs, prev_y, nearby_blocks):

        # Compute Reward
        reward = 0
        
        # Current y position and current block
        agent_y = obs["YPos"]
        cur_block = obs["current_block"][0]
        rstr = "block " + cur_block + ", "
        
        if self.fell_in_water(obs):
            reward -= 300
            rstr += "fell in water -100, "
        else:
            # Check if the agent is at a higher y position than it was before
            if agent_y > prev_y:
                reward += 20
                rstr += "y " + str(agent_y) + " +20, "
            else:
                rstr += "y " + str(agent_y) + " -20, "
        
        # Award agent based on current block they are standing on
        if cur_block =='purpur_slab': # usually does not register because agent is in air
            reward += 50
            rstr += "ppslab +50, "
        elif cur_block =='stone_slab':  # usually does not register because agent is in air
            reward += 250
            rstr += "stone_slab +250, "
        elif cur_block =='gold_block':
            reward += 500 # goal reached
            rstr += "gold +500, "
        elif cur_block == 'air' and agent_y >= 4:
            # If agent is in the air, reward based on height
            reward += 2 ** math.floor(agent_y) * 10  # exponential
            rstr += "air " + "+" + str(2 ** math.floor(agent_y) * 10) + ", "
        
        # Reward agent for being close to gold block
        if any('gold_block' in layer for layer in nearby_blocks):
            reward += 250
            rstr += "close to gold +250, "
        
        # Award for being on/nearby a slab if not in water/bottom
        if math.floor(agent_y) >= 4:
            nearby_slabs = 0
            for i in range(3):
                # print(nearby_blocks[i])
                for j in range(len(nearby_blocks[i])):
                    if nearby_blocks[i][j] in ['stone_slab', 'purpur_slab', 'gold_block']:
                        nearby_slabs += 1
                        rstr += "near slab +10, "
            reward += 10 * nearby_slabs 
        else:
            # Useful when we do not restart mission after agent falls in the water
            distance = self.dist_from_start((obs["XPos"], obs["YPos"],obs["ZPos"]))
            reward -= 10 * 3 ** distance
            rstr += "far from start " + str(10 * 3 ** distance) + ", "
        
        # Penalize agent for the more steps they take by some weight * number of actions taken
#         reward -= (self.num_actions_taken * 0.75)
        
        # accumulate reward for overall mission
        self.reward += reward
        print(rstr)
        
        return reward

    def dist_from_start(self, state):
        goal_x, goal_y, goal_z = 4,3,3
        x, y, z = state
        # manhattan_dist = abs(goal_x - x) + abs(goal_y - y) + abs(goal_z - z)
        manhattan_dist = abs(goal_x - x)
        return round(manhattan_dist)
        
    
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
    
    # load world from xml
#     mission_file = 'finalworld.xml'
#     with open(mission_file, 'r') as f:
#         print("Loading mission from %s" % mission_file)
#         mission_xml = f.read()

    episodes = 5
    prisoner = Prisoner(alpha=0.30)
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
        timedout = False
        
        local_time = time.localtime()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time) + " (UTC)"
        print("Start Time:", formatted_time)
        start = time.time()
        
        # Main loop
        while world_state.is_mission_running:
            
#             Check Timeout
            if (time.time() - start) > prisoner.TIMEOUT:
                print("\nTime limit exceeded. Ending mission.")
                agent_host.sendCommand("quit")  # Exit mission
                timedout = True
                break
            
            world_state = agent_host.getWorldState()
            if world_state.number_of_observations_since_last_state > 0:
                obs_text = world_state.observations[-1].text
                obs = json.loads(obs_text)
                state = prisoner.get_state(obs)
                
                if prev_y is None:
                    prev_y = state[1]

                action = prisoner.choose_action(state)
                if type(action) != str:
                    for command in action:
                        print("COMMAND: " + command)
                        agent_host.sendCommand(command)
                        time.sleep(0.2)
                else:
                    agent_host.sendCommand(action)

                prisoner.inc_num_actions_taken()

                time.sleep(1)

                # Get updated world state and observation
                world_state = agent_host.getWorldState()
                if world_state.number_of_observations_since_last_state > 0:
                    new_obs_text = world_state.observations[-1].text
                    new_obs = json.loads(new_obs_text)

                    new_state = prisoner.get_state(new_obs)
                    reward = prisoner.calc_reward(new_obs, prev_y, new_state[4])
                
                if prev_state != None and prev_action != None:
                    prisoner.update_q(prev_state, prev_action, reward, state)
                
                prev_state = state
                prev_action = action
                prev_y = state[1]
                
                # Every once in a while, print the Q table
#                 if prisoner.get_num_actions_taken() % 1000 == 0:
#                     print("Current Q-table:", prisoner.q_table)
               
                
        end = time.time()
        length = end - start
        local_time = time.localtime()
        formatted_time = formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time) + " (UTC)"
        print("End Time:", formatted_time)
        print("Time elapsed:", length, "seconds")
        if not timedout:
            print("Goal reached!")
        
        # At the end of each episode, reduce the epsilon
        if prisoner.epsilon > prisoner.epsilon_min:
            prisoner.epsilon *= prisoner.epsilon_decay
            prisoner.epsilon = max(prisoner.epsilon, prisoner.epsilon_min)
        print("Epsilon after episode {}: {:.4f}".format(episode + 1, prisoner.epsilon))

print("Training complete!")