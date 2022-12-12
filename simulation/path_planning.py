from enum import Enum
import recognition
import networkx as nx
import math
import numpy as np

class PathPlanningAlgorithm:
    DWA = 0
    ASTAR = 1
    BUG = 2

    def __init__(self, scene) -> None:
        self.scene = scene
    def nextMove(self,current_pos=None):
        pass
    def reachedGoal(self):
        pass

class RobotType(Enum):
    circle = 0
    rectangle = 1

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
        self.robot_stuck_flag_cons = 0.01  # constant to prevent robot stucked
        self.robot_type = RobotType.circle

        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.robot_radius = 1.1  # [m] for collision check

        # if robot_type == RobotType.rectangle
        self.robot_width = 0.5  # [m] for collision check
        self.robot_length = 1.2  # [m] for collision check


class DynamicWindowAlgorithm(PathPlanningAlgorithm):
    def __init__(self, scene, init_state=[0,0,math.pi / 8.0, 0.0, 0.0]) -> None:
        super().__init__(scene)
        self.robot_config = RobotConfig()
        self.state = init_state
        self.state[0] = scene.source_loc[0]
        self.state[1] = scene.source_loc[1]

    def dwa_control(self, x):
        """
        Dynamic Window Approach control
        """
        dw = self.calc_dynamic_window(x)

        u, trajectory = self.calc_control_and_trajectory(x, dw)

        return u, trajectory

    def motion(self, x, u):
        """
        motion model
        """
        dt = self.robot_config.dt
        x[2] += u[1] * dt
        x[0] += u[0] * math.cos(x[2]) * dt
        x[1] += u[0] * math.sin(x[2]) * dt
        x[3] = u[0]
        x[4] = u[1]

        return x


    def calc_dynamic_window(self, x):
        """
        calculation dynamic window based on current state x
        """
        config = self.robot_config
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


    def predict_trajectory(self, x_init, v, y):
        """
        predict trajectory with an input
        """
        config = self.robot_config
        x = np.array(x_init)
        trajectory = np.array(x)
        time = 0
        while time <= config.predict_time:
            x = self.motion(x, [v, y])
            trajectory = np.vstack((trajectory, x))
            time += config.dt

        return trajectory


    def calc_control_and_trajectory(self, x, dw):
        """
        calculation final input with dynamic window
        """
        config = self.robot_config
        x_init = x[:]
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_trajectory = np.array([x])

        # evaluate all trajectory with sampled input in dynamic window
        for v in np.arange(dw[0], dw[1], config.v_resolution):
            for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):

                trajectory = self.predict_trajectory(x_init, v, y)
                # calc cost
                to_goal_cost = config.to_goal_cost_gain * self.calc_to_goal_cost(trajectory)
                speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
                ob_cost = config.obstacle_cost_gain * self.calc_obstacle_cost(trajectory)

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


    def calc_obstacle_cost(self, trajectory):
        """
        calc obstacle cost inf: collision
        """
        config = self.robot_config
        ob = np.array(self.scene.ob)
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


    def calc_to_goal_cost(self, trajectory):
        """
            calc to goal cost with angle difference
        """
        goal = self.scene.goal_loc
        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        error_angle = math.atan2(dy, dx)
        cost_angle = error_angle - trajectory[-1, 2]
        cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

        return cost

    def nextMove(self, current_pos=None):
        # if current_pos is not None:
        #     self.state[0] = current_pos[0]
        #     self.state[1] = current_pos[1]
        u, predicted_trajectory = self.dwa_control(self.state)
        self.state = self.motion(self.state,u)
        return self.state[:2]

    def reachedGoal(self):
        dist_to_goal = math.hypot(self.state[0]-self.scene.goal_loc[0], self.state[1]-self.scene.goal_loc[1])
        return dist_to_goal <= self.robot_config.robot_radius
        



class AStarPlanningAlgorithm(PathPlanningAlgorithm):
    def __init__(self, scene) -> None:
        super().__init__(scene)
        self.network = self._construct_network()
        self.trajectory = self._get_trajectory()
        self.state = 0


    def _get_trajectory(self):
        origin = tuple(self.scene.source_loc)
        destination = tuple(self.scene.goal_loc)
        return nx.astar_path(self.network,origin,destination)



    def _construct_network(self):
        G = recognition.construct_euclidean_network(self.scene)
        obstructed_G = recognition.apply_obstacles_network(self.scene,G)
        return obstructed_G

    def nextMove(self, currect_pos=None):
        pos = self.trajectory[self.state]
        self.state = (self.state + 1) % len(self.trajectory)
        return pos 

    def reachedGoal(self):
        return self.state == len(self.trajectory) - 1

class BugPlanner(PathPlanningAlgorithm):
    def __init__(self, scene):
        source = scene.source_loc
        goal = scene.goal_loc
        obs = scene.ob
        self.goal = tuple(goal)
        self.obs = [tuple(ob) for ob in obs]
        self.r = [tuple(source)]
        self.out = []
        self._set_exterior_points()
        self.visited = []
        

    def _set_exterior_points(self):
        for o_x, o_y in self.obs:
            for add_x in [-1,0,1]:
                for add_y in [-1,0,1]:
                    cand = (o_x+add_x, o_y+add_y)
                    if cand not in self.obs:
                        self.out.append(cand)

    def reachedGoal(self):
        return self.r[-1] == self.goal

    def mov_normal(self):
        return self.r[-1][0] + np.sign(self.goal[0] - self.r[-1][0]), \
               self.r[-1][1] + np.sign(self.goal[1] - self.r[-1][1])

    def mov_to_next_obs(self):
        visited = self.visited
        for add_x, add_y in zip([1, 0, -1, 0], [0, 1, 0, -1]):
            cand = self.r[-1][0] + add_x, self.r[-1][1] + add_y
            if cand in self.out and cand not in visited:
                return cand, False
        return self.r[-1], True


class Bug0Planner(BugPlanner):
    """
        Greedy algorithm where you move towards goal
        until you hit an obstacle. Then you go around it
        (pick an arbitrary direction), until it is possible
        for you to start moving towards goal in a greedy manner again
        """

    def nextMove(self, current_pos=None):
        cand = self.mov_normal()
        if cand in self.obs:
            cand, _ = self.mov_to_next_obs()
            self.visited.append(cand)
        else:
            self.visited = []
        self.r.append(cand)
        return cand
