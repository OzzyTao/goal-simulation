import networkx as nx
import numpy as np
from math import sqrt
from itertools import combinations

NEG_SLOPE_WEIGHT = 0.5


class MastersGoalRecognition:
    def __init__(self, source, goals):
        self.goals = goals
        self.trajectory = [tuple(source)]
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
                goal.masters_rank = self.ranking[i]
        self.trajectory.append(observation)

class MirroringGoalRecognition:
    def __init__(self, source, goals) -> None:
        self.goals = goals 
        self.prefix_costs = np.zeros(len(goals))
        self.trajectory = [tuple(source)]
        self.ranking = np.array(range(len(goals)))
        self.optimal_costs = list(map(self._calculate_optimal_cost,goals))

    def _calculate_optimal_cost(self, goal):
        return goal.cost_func(self.trajectory[0],goal.dest_pos)

    def _update_prefix(self):
        self.prefix_costs = self.prefix_costs + np.array([goal.cost_func(self.trajectory[-2],self.trajectory[-1]) for goal in self.goals])

    def _ranks(self):
        scores = []
        for i, goal in enumerate(self.goals):
            optimal_cost = self.optimal_costs[i]
            prefix_cost = self.prefix_costs[i]
            surfix_cost = goal.cost_func(self.trajectory[-1],goal.dest_pos)
            scores.append(optimal_cost/(prefix_cost+surfix_cost))
        order = (-np.array(scores)).argsort()
        return order.argsort()

    def step(self, observation):
        self.trajectory.append(observation)
        self._update_prefix()
        self.ranking = self._ranks()
        for i, goal in enumerate(self.goals):
            goal.mirroring_rank = self.ranking[i]


class OnlineGoalRecognition:
    def __init__(self, source, goals):
        # each goal has a destination and a cost function
        self.goals = goals

        self.current_segment = [tuple(source)]
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
    FULLY_CONNECTED_SIZE = 1
    obs = [tuple(ob) for ob in scene_config.ob]
    G = nx.grid_2d_graph(scene_config.width, scene_config.height)
    for n in G:
        for dx in range(-FULLY_CONNECTED_SIZE,FULLY_CONNECTED_SIZE+1):
            for dy in range(-FULLY_CONNECTED_SIZE,FULLY_CONNECTED_SIZE+1):
                target = (n[0]+dx, n[1]+dy)
                if 0<=target[0]<scene_config.width and 0<=target[1]<scene_config.height:
                    weight = sqrt(dx**2 + dy**2)
                    G.add_edge(n,target,weight=weight)
    return G

# def construct_full_network(scene_config):
#     G = nx.Graph()
#     nodes = []
#     for i in range(scene_config.width):
#         for j in range(scene_config.height):
#             nodes.append((i,j))
#     edges = combinations(nodes, 2)
#     weighted_edges = list(map(lambda e: (e[0],e[1],eu_dist(e[0],e[1])),edges))
#     G.add_nodes_from(nodes)
#     G.add_weighted_edges_from(weighted_edges)
#     return G


def apply_obstacles_network(scene_config, network):
    G = network.copy()
    obs = [tuple(ob) for ob in scene_config.ob]
    for u,v in G.edges(obs):
        G[u][v]['weight'] = 10000
    return G

# def eu_dist(a,b):
#     (x1,y1) = a
#     (x2,y2) = b
#     return sqrt((x1-x2)**2+(y1-y2)**2)

# def calculate_cost_network(network, source, target):
#     return nx.astar_path_length(network,source,target,heuristic=eu_dist,weight="weight")
