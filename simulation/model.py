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



class Robot(mesa.Agent):
    def __init__(self, unique_id, model, alg_cls):
        super(Robot, self).__init__(unique_id, model)
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
    def __init__(self, unique_id, model, dest_pos, cost_func):
        super().__init__(unique_id, model)
        self.cost_func = cost_func
        self.dest_pos = dest_pos
        self.rank = unique_id
        self.benchmark_rank = unique_id


# def get_total_rankings(model):
#     return [goal.rank for goal in model.prediction_goals]

# def get_benchmark_rankings(model):
#     return [goal.benchmark_rank for goal in model.prediction_goals]

def get_agent_cost(agent):
    if isinstance(agent,Goal):
        cur_pos = agent.model.robot.state
        pos = (int(cur_pos[0]),int(cur_pos[1]))
        return agent.cost_func(pos,agent.dest_pos)
    else:
        return None

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
    def __init__(self, width, height, obs_num, goals_num, path_planning_alg = 2, goal_zone=0, seed=None):
        ## config experiment scene
        fscene = scene.FixedScene(self.random,width,height,int(math.sqrt(obs_num)), int(math.sqrt(obs_num)))
        fscene.set_destination(goal_zone)
        fscene.set_candidate_desinations(goals_num,goal_zone)
        scene_config = fscene.scene
        self.scene_config = scene_config
        self.grid = mesa.space.MultiGrid(width,height,True)

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
        cost_funcs = [cost_func_direct, cost_func_avoid]
        self.prediction_goals = []
        for i, goal in enumerate(self.prediction_dests):
            for j, cost_func in enumerate(cost_funcs):
                goal_agent = Goal(i*len(cost_funcs)+j,self,tuple(goal),cost_func)
                self.grid.place_agent(goal_agent,tuple(goal))
                self.prediction_goals.append(goal_agent)
        
        ## install all goal recognition models of interest
        self.recognition_models = {'Segment':recognition.OnlineGoalRecognition(scene_config.source_loc,self.prediction_goals),
                                    'Masters':recognition.MastersGoalRecognition(scene_config.source_loc, self.prediction_goals),
                                    'Mirroring':recognition.MirroringGoalRecognition(scene_config.source_loc, self.prediction_goals)}
        
        ## variable reportors
        model_reportor = {'seed':"_seed"}
        for k in self.recognition_models:
            model_reportor[k+'_ranking'] = self.get_recognition_ranking(k)
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
            self.recognition_models[k].step(loc)
        self.datacollector.collect(self)

    def get_recognition_ranking(self, model_name):
        recognition_model = self.recognition_models[model_name]
        def tmp(model):
            return recognition_model.ranking.tolist()
        return tmp

