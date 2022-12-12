import mesa
import math
from enum import Enum
import numpy as np
from random import randrange
from recognition import OnlineGoalRecognition, construct_euclidean_network, calculate_cost_network, eu_dist, BenchmarkGoalRecognition

class RobotType(Enum):
    circle = 0
    rectangle = 1

class SceneConfig:
    def __init__(self,width, height,ob_points, source_loc, goal_loc):
        self.ob = ob_points
        self.source_loc = source_loc
        self.goal_loc = goal_loc
        self.width = width
        self.height = height

def create_random_scene(width, height, ob_patch_num):
    obs = []
    to_visit_stack = []
    while len(obs)<ob_patch_num:
        if len(to_visit_stack)==0:
            to_visit_stack.append([randrange(width),randrange(height)])
        current_idx = randrange(len(to_visit_stack))
        current_point = to_visit_stack.pop(current_idx)
        obs.append(current_point)
        for dx,dy in [[0,-1],[0,1],[-1,0],[1,0]]:
            if (0 <= dx+current_point[0] < width) and (0 <= dy+current_point[1] < height):
                tmp_loc = [dx+current_point[0],dy+current_point[1]]
                if tmp_loc not in to_visit_stack and tmp_loc not in obs:
                    to_visit_stack.append(tmp_loc)
    source = []
    while len(source)==0:
        tmp = [randrange(width),randrange(height)]
        if tmp not in obs:
            source = tmp
    target = []
    while len(target)==0:
        tmp = [randrange(width),randrange(height)]
        if tmp not in obs and tmp != source:
            target = tmp
    return SceneConfig(width,height,obs,source,target)

class RobotConfig:
    def __init__(self):
        # robot parameter
        self.max_speed = 1.0  # [m/s]
        self.min_speed = -0.5  # [m/s]
        self.max_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 0.2  # [m/ss]
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/ss]
        self.v_resolution = 0.01  # [m/s]
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0  # [rad/s]
        self.dt = 0.2  # [s] Time tick for motion prediction
        self.predict_time = 3.0  # [s]
        self.to_goal_cost_gain = 0.15
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 1.0
        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked
        self.robot_type = RobotType.circle

        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.robot_radius = 1.0  # [m] for collision check

        # if robot_type == RobotType.rectangle
        self.robot_width = 0.5  # [m] for collision check
        self.robot_length = 1.2  # [m] for collision check


def dwa_control(x, config, goal, ob):
    """
    Dynamic Window Approach control
    """
    dw = calc_dynamic_window(x, config)

    u, trajectory = calc_control_and_trajectory(x, dw, config, goal, ob)

    return u, trajectory

def motion(x, u, dt):
    """
    motion model
    """

    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]

    return x


def calc_dynamic_window(x, config):
    """
    calculation dynamic window based on current state x
    """

    # Dynamic window from robot specification
    Vs = [config.min_speed, config.max_speed,
          -config.max_yaw_rate, config.max_yaw_rate]

    # Dynamic window from motion model
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_delta_yaw_rate * config.dt,
          x[4] + config.max_delta_yaw_rate * config.dt]

    #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    return dw


def predict_trajectory(x_init, v, y, config):
    """
    predict trajectory with an input
    """

    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)
        trajectory = np.vstack((trajectory, x))
        time += config.dt

    return trajectory


def calc_control_and_trajectory(x, dw, config, goal, ob):
    """
    calculation final input with dynamic window
    """

    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])

    # evaluate all trajectory with sampled input in dynamic window
    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):

            trajectory = predict_trajectory(x_init, v, y, config)
            # calc cost
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
            speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
            ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, ob, config)

            final_cost = to_goal_cost + speed_cost + ob_cost

            # search minimum trajectory
            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v, y]
                best_trajectory = trajectory
                if abs(best_u[0]) < config.robot_stuck_flag_cons \
                        and abs(x[3]) < config.robot_stuck_flag_cons:
                    # to ensure the robot do not get stuck in
                    # best v=0 m/s (in front of an obstacle) and
                    # best omega=0 rad/s (heading to the goal with
                    # angle difference of 0)
                    best_u[1] = -config.max_delta_yaw_rate
    return best_u, best_trajectory


def calc_obstacle_cost(trajectory, ob, config):
    """
    calc obstacle cost inf: collision
    """
    ox = ob[:, 0]
    oy = ob[:, 1]
    dx = trajectory[:, 0] - ox[:, None]
    dy = trajectory[:, 1] - oy[:, None]
    r = np.hypot(dx, dy)

    if config.robot_type == RobotType.rectangle:
        yaw = trajectory[:, 2]
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        rot = np.transpose(rot, [2, 0, 1])
        local_ob = ob[:, None] - trajectory[:, 0:2]
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        local_ob = np.array([local_ob @ x for x in rot])
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        upper_check = local_ob[:, 0] <= config.robot_length / 2
        right_check = local_ob[:, 1] <= config.robot_width / 2
        bottom_check = local_ob[:, 0] >= -config.robot_length / 2
        left_check = local_ob[:, 1] >= -config.robot_width / 2
        if (np.logical_and(np.logical_and(upper_check, right_check),
                           np.logical_and(bottom_check, left_check))).any():
            return float("Inf")
    elif config.robot_type == RobotType.circle:
        if np.array(r <= config.robot_radius).any():
            return float("Inf")

    min_r = np.min(r)
    return 1.0 / min_r  # OK


def calc_to_goal_cost(trajectory, goal):
    """
        calc to goal cost with angle difference
    """

    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    error_angle = math.atan2(dy, dx)
    cost_angle = error_angle - trajectory[-1, 2]
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

    return cost

def generate_random_goals(scene_config, num):
    prediction_goals = []
    avoid_poses = [scene_config.source_loc,scene_config.goal_loc]
    while len(prediction_goals) < num:
        tmp_pos = [randrange(scene_config.width),randrange(scene_config.height)]
        if tmp_pos not in avoid_poses:
            prediction_goals.append(tmp_pos)
            avoid_poses.append(tmp_pos)
    return prediction_goals

class Robot(mesa.Agent):
    def __init__(self, unique_id, model, config, state, goal):
        super(Robot, self).__init__(unique_id, model)
        self.config = config
        self.state = state
        self.trajectory = np.array(state)
        self.goal = goal

    def move(self):
        u, predicted_trajectory = dwa_control(self.state, self.config, self.goal, np.array(self.model.scene_config.ob))
        tmp_state = motion(self.state, u, self.config.dt)
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
        dist_to_goal = math.hypot(self.state[0]-self.goal[0], self.state[1]-self.goal[1])
        return dist_to_goal <= self.config.robot_radius

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


def get_total_rankings(model):
    return [goal.rank for goal in model.prediction_goals]

def get_benchmark_rankings(model):
    return [goal.benchmark_rank for goal in model.prediction_goals]

def get_agent_cost(agent):
    if isinstance(agent,Goal):
        cur_pos = agent.model.robot.state
        pos = (int(cur_pos[0]),int(cur_pos[1]))
        return agent.cost_func(pos,agent.dest_pos)
    else:
        return None

def get_agent_rank(agent):
    if isinstance(agent,Goal):
        return agent.rank
    else:
        return None

def get_agent_benchmark_rank(agent):
    if isinstance(agent,Goal):
        return agent.benchmark_rank
    else:
        return None

class PathFindingModel(mesa.Model):
    def __init__(self, width, height, obs_num, goals_num):
        scene_config = create_random_scene(width,height,obs_num)
        robot_config = RobotConfig()
        self.scene_config = scene_config
        self.grid = mesa.space.MultiGrid(width,height,True)
        self.schedule = mesa.time.RandomActivation(self)

        # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
        self.robot = Robot(2000,self,robot_config,
                           [scene_config.source_loc[0],scene_config.source_loc[1],math.pi / 8.0, 0.0, 0.0],
                           scene_config.goal_loc)
        self.grid.place_agent(self.robot,tuple(scene_config.source_loc))
        self.schedule.add(self.robot)
        # place static objects on the grid
        for i, ob in enumerate(scene_config.ob):
            ob_agent = Obstacle(1000+i,self)
            self.grid.place_agent(ob_agent, tuple(ob))
        #
        # # place goal
        # goal = Goal(0, self)
        # pos = (np.int64(scene_config.goal_loc[0]), np.int64(scene_config.goal_loc[1]))
        # self.grid.place_agent(goal, pos)
        #
        ## goal recognition
        G = construct_euclidean_network(scene_config)
        self.prediction_dests = [scene_config.goal_loc] + generate_random_goals(scene_config, goals_num)
        cost_func_avoid = lambda x,y: calculate_cost_network(G,x,y)
        cost_funcs = [cost_func_avoid,eu_dist]
        self.prediction_goals = []
        for i, goal in enumerate(self.prediction_dests):
            for j, cost_func in enumerate(cost_funcs):
                goal_agent = Goal(i*len(cost_funcs)+j,self,tuple(goal),cost_func)
                self.grid.place_agent(goal_agent,tuple(goal))
                self.prediction_goals.append(goal_agent)

        self.recognition_model = OnlineGoalRecognition(self,self.prediction_goals)
        self.benchmark_recognition_model = BenchmarkGoalRecognition(self, self.prediction_goals)

        ## variable reporters
        self.datacollector = mesa.DataCollector(
            model_reporters={'Goal_Rank':get_total_rankings, 'Benchmark_Rank':get_benchmark_rankings},
        )
        # agent_reporters = {'cost': get_agent_cost, 'rank': get_agent_rank}
        self.running = True


    def step(self):
        self.schedule.step()
        loc = (int(self.robot.state[0]),int(self.robot.state[1]))
        self.recognition_model.step(loc)
        self.benchmark_recognition_model.step(loc)
        self.datacollector.collect(self)

