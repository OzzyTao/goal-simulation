import networkx as nx
import numpy as np
from math import sqrt

NEG_SLOPE_WEIGHT = 0.5

class BenchmarkGoalRecognition:
    def __init__(self,model,goals):
        self.model = model
        self.goals = goals
        self.trajectory = []
        self.ranking = np.array(range(len(goals)))

    def _slope(self, start, end, goal):
        return goal.cost_func(start, goal.dest_pos)-goal.cost_func(end,goal.dest_pos)

    def _slopes(self, start, end):
        return np.array([self._slope(start,end,goal) for goal in self.goals])

    def _slope_ranks(self, start, end):
        slopes = self._slopes(start, end)
        order = (-slopes).argsort()
        ranks = order.argsort()
        return ranks

    def step(self, observation):
        if self.trajectory:
            start = self.trajectory[-1]
            self.ranking = self._slope_ranks(start,observation)
            for i, goal in enumerate(self.goals):
                goal.benchmark_rank = self.ranking[i]
        self.trajectory.append(observation)




class OnlineGoalRecognition:
    def __init__(self, model, goals):
        # each goal has a destination and a cost function
        self.model = model
        self.goals = goals

        self.current_segment = []
        self.slope_ranking = np.array(range(len(goals)))

        self.segments = []
        self.ranking = np.array(range(len(goals)))

    def _segment(self,observation):
        state = len(self.current_segment)
        if state==0:
            self.current_segment.append(observation)
        elif state==1:
            self.current_segment.append(observation)
            self.slope_ranking = self._slope_ranks(self.current_segment[0], self.current_segment[1])
        else:
            if np.array_equal(self._slope_ranks(self.current_segment[0],observation), self.slope_ranking):
                self.current_segment.append(observation)
            else:
                self.segments.append(self.current_segment)
                self.current_segment = [self.current_segment[-1],observation]
                self.slope_ranking = self._slope_ranks(self.current_segment[0],self.current_segment[1])

    def _slope(self, start, end, goal):
        return goal.cost_func(start, goal.dest_pos)-goal.cost_func(end,goal.dest_pos)

    def _slopes(self, start, end):
        return np.array([self._slope(start,end,goal) for goal in self.goals])

    def _slope_ranks(self, start, end):
        slopes = self._slopes(start, end)
        order = (-slopes).argsort()
        ranks = order.argsort()
        return ranks

    def _goal_rank_score(self, goal):
        score = 0
        lengths = [len(seg)-1 for seg in self.segments]
        if self.current_segment:
            for i, seg in enumerate(self.segments):
                length = lengths[i]
                slope = self._slope(seg[0],seg[-1],goal)/length
                recency = sum(lengths[i:])
                tmp_func = lambda x: 1 if x>0 else NEG_SLOPE_WEIGHT
                h = tmp_func(slope)
                score += length*slope/(recency*h)
            length = len(self.current_segment)
            slope = self._slope(self.current_segment[0],self.current_segment[-1],goal)/length
            recency = 1
            h = 1 if slope > 0 else NEG_SLOPE_WEIGHT
            score += length * slope / (recency * h)
        return score

    def _goal_ranks(self):
        scores = np.array([self._goal_rank_score(goal) for goal in self.goals])
        order = (-scores).argsort()
        ranks = order.argsort()
        return ranks

    def step(self, observation):
        self._segment(observation)
        #update goal object
        self.ranking = self._goal_ranks()
        for i,goal in enumerate(self.goals):
            goal.rank = self.ranking[i]

def construct_euclidean_network(scene_config):
    obs = [tuple(ob) for ob in scene_config.ob]
    G = nx.grid_2d_graph(scene_config.width, scene_config.height)
    for n in G:
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                target = (n[0]+dx, n[1]+dy)
                if 0<=target[0]<scene_config.width and 0<=target[1]<scene_config.height:
                    weight = sqrt(dx**2 + dy**2)
                    if (n in obs) or (target in obs):
                        weight = 1000
                    G.add_edge(n,target,weight=weight)
    return G

def apply_obstacles_network(scene_config, network):
    nodes = network.nodes()
    obs = [tuple(ob) for ob in scene_config.ob]
    G =  network.subgraph([node for node in nodes if node not in obs])
    return G

def eu_dist(a,b):
    (x1,y1) = a
    (x2,y2) = b
    return sqrt((x1-x2)**2+(y1-y2)**2)

def calculate_cost_network(network, source, target):
    return nx.astar_path_length(network,source,target,heuristic=eu_dist,weight="weight")
