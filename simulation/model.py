import os, sys, inspect
# realpath() will make your script run, even if you symlink it :)
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

import mesa
import math
import numpy as np
import recognition
import scene
import path_planning as pp
import cost_funcs as cf
import time



class Robot(mesa.Agent):
    def __init__(self, unique_id, model, alg_cls):
        super(Robot, self).__init__(unique_id, model)
        ## state is a tuple (x,y) representing the current location of the robot
        state = model.scene_config.source_loc
        self.state = state
        self.trajectory = np.array(state)
        self.goal = model.scene_config.goal_loc
        # dynamic window approach for path planning
        # self.algorithm = pp.DynamicWindowAlgorithm(self.model.scene_config,state)
        # A star for path planning
        self.algorithm  = alg_cls(self.model.scene_config)

    def move(self):
        tmp_state = self.algorithm.nextMove(self.state)
        if tmp_state[0] < 0:
            tmp_state[0] = 0
        if tmp_state[0] > self.model.scene_config.width -1 :
            tmp_state[0] = self.model.scene_config.width - 1
        if tmp_state[1] < 0:
            tmp_state[1] = 0
        if tmp_state[1] > self.model.scene_config.height - 1:
            tmp_state[1] = self.model.scene_config.height - 1
        self.state = tmp_state
        self.trajectory = np.vstack((self.trajectory,self.state))
        pos = (np.int64(self.state[0]),np.int64(self.state[1]))
        self.model.grid.move_agent(self,pos)

    def reached_goal(self):
        return self.algorithm.reachedGoal()

    def step(self):
        if not self.reached_goal():
            self.move()
        else:
            self.model.running = False

class Obstacle(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.cost = 100

class Goal(mesa.Agent):
    def __init__(self, unique_id, model, dest_pos, cost_func, ground_truth = False):
        super().__init__(unique_id, model)
        self.cost_func = cost_func
        self.dest_pos = dest_pos
        self.rank = unique_id
        self.benchmark_rank = unique_id
        self.ground_truth = ground_truth


# def get_total_rankings(model):
#     return [goal.rank for goal in model.prediction_goals]

# def get_benchmark_rankings(model):
#     return [goal.benchmark_rank for goal in model.prediction_goals]

def get_segment_number(model):
    return len(model.recognition_models['Segment'].segments)

def get_agent_cost(agent):
    if isinstance(agent,Goal):
        cur_pos = agent.model.robot.state
        pos = (int(cur_pos[0]),int(cur_pos[1]))
        return agent.cost_func(pos,agent.dest_pos)
    else:
        return None

def is_obstacle_used(cost_network_obstacle, cost_network, scene_config):
    source = tuple(scene_config.source_loc)
    goal = tuple(scene_config.goal_loc)
    cost_with_obstacle = cost_network_obstacle(source,goal)
    cost_without_obstacle = cost_network(source,goal)
    return cost_with_obstacle != cost_without_obstacle

# def get_agent_rank(agent):
#     if isinstance(agent,Goal):
#         return agent.rank
#     else:
#         return None

# def get_agent_benchmark_rank(agent):
#     if isinstance(agent,Goal):
#         return agent.benchmark_rank
#     else:
#         return None

class PathFindingModel(mesa.Model):
    def __init__(self, width, height, obs_num, goal_zones='0,0', path_planning_alg = 2, intention_profile=0, seed=None):
        '''
        width: width of the scene
        height: height of the scene
        obs_num: number of obstacles of the scene
        goal_zones: an integer or a list with the length of goals_num, indicating the zone of each goal
        path_planning_alg: path planning algorithm for movement simulation
        intention_profile: 0 for single objective intention and 1 for multi-objective intention
        '''
        ## config experiment scene
        fscene = scene.FixedScene(self.random,width,height,int(math.sqrt(obs_num)), int(math.sqrt(obs_num)))
        zones = goal_zones.split(',')
        fscene.set_destination(zones[0])
        fscene.set_candidate_desinations(zones[1:] if len(zones) > 1 else [])
        scene_config = fscene.scene
        self.scene_config = scene_config
        self.grid = mesa.space.MultiGrid(width,height,True)

        ### random scene
        # fscene = scene.RandomScene(self.random,width,height,obs_num)
        # fscene.add_candidate_destinations(goals_num)
        # scene_config = fscene.scene
        # self.scene_config = scene_config
        # self.grid = mesa.space.MultiGrid(width,height,True)

        ## config agent and its path planning algorithm
        alg_cls = None
        if path_planning_alg == pp.PathPlanningAlgorithm.DWA:
            alg_cls = pp.DynamicWindowAlgorithm
        elif path_planning_alg == pp.PathPlanningAlgorithm.ASTAR:
            alg_cls = pp.AStarPlanningAlgorithm
        else:
            alg_cls = pp.Bug0Planner
        self.robot = Robot(2000,self,alg_cls)
        
        ## goal recognition: define cost functions for each goal


        self.prediction_dests = [scene_config.goal_loc] + fscene.destinations
        cost_func_direct = cf.NetworkCost(scene_config,apply_obstacls=False)
        cost_func_avoid = cf.NetworkCost(scene_config,apply_obstacls=True)
        cost_funcs = [cost_func_avoid,cost_func_direct] if intention_profile else [cost_func_direct]
        self.prediction_goals = []
        for i, goal in enumerate(self.prediction_dests):
            for j, cost_func in enumerate(cost_funcs):
                if i == 0 and j == 0:
                    goal_agent = Goal(i*len(cost_funcs)+j,self,tuple(goal),cost_func,ground_truth=True)
                else:
                    goal_agent = Goal(i*len(cost_funcs)+j,self,tuple(goal),cost_func)
                self.grid.place_agent(goal_agent,tuple(goal))
                self.prediction_goals.append(goal_agent)
        self.random.shuffle(self.prediction_goals)
        self.intention_num = len(self.prediction_goals)
        for index, goal in enumerate(self.prediction_goals):
            if goal.ground_truth:
                self.true_intention = index
        self.obstacle_used = is_obstacle_used(cost_func_avoid,cost_func_direct,scene_config)
        
        ## install all goal recognition models of interest
        self.recognition_models = {'Segment':recognition.OnlineGoalRecognition(scene_config.source_loc,self.prediction_goals),
                                    'Masters':recognition.MastersGoalRecognition(scene_config.source_loc, self.prediction_goals),
                                    'Mirroring':recognition.MirroringGoalRecognition(scene_config.source_loc, self.prediction_goals)}
        #  'FastSegment':recognition.FastIntentionRecognition(scene_config.source_loc,self.prediction_goals),
        
        ## variable reportors
        model_reportor = {'seed':"_seed", 'true_intention':"true_intention", 'intention_num':"intention_num", 'obstacle_used':"obstacle_used", 'segment_num':get_segment_number}
        for k in self.recognition_models:
            model_reportor[k+'_ranking'] = self.get_recognition_ranking(k)
            model_reportor[k+'_probs'] = self.get_recognition_probs(k)
            # model_reportor[k+'_step_time'] = self.get_recognition_step_time(k)
        self.datacollector = mesa.DataCollector(
            model_reporters=model_reportor
        )
        ## variable reporters
        # self.datacollector = mesa.DataCollector(
        #     model_reporters={'Goal_Rank':get_total_rankings, 
        #     'Benchmark_Rank':get_benchmark_rankings, 
        #     'seed':"_seed", 
        #     },
        # )
        # agent_reporters = {'cost': get_agent_cost, 'rank': get_agent_rank}
        
        ## create mesa scene, agents
        self.schedule = mesa.time.RandomActivation(self)
        self.grid.place_agent(self.robot,tuple(scene_config.source_loc))
        self.schedule.add(self.robot)
        # place static objects on the grid
        for i, ob in enumerate(scene_config.ob):
            ob_agent = Obstacle(1000+i,self)
            self.grid.place_agent(ob_agent, tuple(ob))
        self.running = True


    def step(self):
        self.schedule.step()
        loc = (np.int64(self.robot.state[0]),np.int64(self.robot.state[1]))
        for k in self.recognition_models:
            start = time.time()
            self.recognition_models[k].step(loc)
            end = time.time()
            self.recognition_models[k].step_time = end - start
        self.datacollector.collect(self)

    def get_recognition_ranking(self, model_name):
        recognition_model = self.recognition_models[model_name]
        def tmp(model):
            return recognition_model.ranking.tolist()
        return tmp
    
    def get_recognition_probs(self, model_name):
        recognition_model = self.recognition_models[model_name]
        def tmp(model):
            return recognition_model.probs.tolist()
        return tmp
    
    def get_recognition_step_time(self, model_name):
        recognition_model = self.recognition_models[model_name]
        def tmp(model):
            return recognition_model.step_time
        return tmp

