import networkx as nx
import numpy as np
from math import sqrt
from itertools import combinations

NEG_SLOPE_WEIGHT = 0.5

def rank_scores(scores):
    order = (-np.array(scores)).argsort()
    ranks = order.argsort()
    return ranks

class GoalRecognition:
    def __init__(self) -> None:
        self.step_time = 0

class MastersGoalRecognition(GoalRecognition):
    def __init__(self, source, goals):
        super().__init__()
        self.goals = goals
        self.trajectory = [tuple(source)]
        self.ranking = np.array(range(len(goals)))

    def _costdif(self, start, end, goal):
        return goal.cost_func(end,goal.dest_pos) - goal.cost_func(start, goal.dest_pos)
    
    def _probs(self, costdifs, beta=1):
        exps = np.exp(-beta*costdifs)
        values = exps/(1+exps)
        return values/sum(values)

    def _slopes(self, start, end):
        return np.array([self._slope(start,end,goal) for goal in self.goals])

    def step(self, observation):
        if self.trajectory:
            start = self.trajectory[0]
            costdifs = np.array([self._costdif(start,observation,goal) for goal in self.goals])
            self.probs = self._probs(costdifs)
            self.ranking = rank_scores(self.probs)
            for i, goal in enumerate(self.goals):
                goal.masters_rank = self.ranking[i]
                goal.masters_prob = self.probs[i]
        self.trajectory.append(observation)

class MirroringGoalRecognition(GoalRecognition):
    def __init__(self, source, goals) -> None:
        super().__init__()
        self.goals = goals 
        self.prefix_costs = np.zeros(len(goals))
        self.trajectory = [tuple(source)]
        self.ranking = np.array(range(len(goals)))
        self.optimal_costs = list(map(self._calculate_optimal_cost,goals))

    def _calculate_optimal_cost(self, goal):
        return goal.cost_func(self.trajectory[0],goal.dest_pos)

    def _update_prefix(self):
        self.prefix_costs = self.prefix_costs + np.array([goal.cost_func(self.trajectory[-2],self.trajectory[-1]) for goal in self.goals])

    def _scores(self):
        scores = []
        for i, goal in enumerate(self.goals):
            optimal_cost = self.optimal_costs[i]
            prefix_cost = self.prefix_costs[i]
            surfix_cost = goal.cost_func(self.trajectory[-1],goal.dest_pos)
            scores.append(optimal_cost/(prefix_cost+surfix_cost))
        return scores

    def step(self, observation):
        self.trajectory.append(observation)
        self._update_prefix()
        scores = self._scores()
        self.ranking = rank_scores(scores)
        self.probs = np.array(scores)/sum(scores)
        for i, goal in enumerate(self.goals):
            goal.mirroring_rank = self.ranking[i]
            goal.mirroring_prob = self.probs[i]


class OnlineGoalRecognition(GoalRecognition):
    def __init__(self, source, goals):
        # each goal has a destination and a cost function
        super().__init__()
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

    def _goal_rank_score(self, goal_idx):
        goal = self.goals[goal_idx]
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
        scores = np.array([self._goal_rank_score(i) for i in range(len(self.goals))])
        order = (-scores).argsort()
        ranks = order.argsort()
        return ranks

    def step(self, observation):
        self._segment(observation)
        #update goal object
        scores = np.array([self._goal_rank_score(i) for i in range(len(self.goals))])
        self.probs = scores/sum(scores) if sum(scores)!=0 else np.zeros(len(scores))
        self.ranking = rank_scores(scores)
        for i,goal in enumerate(self.goals):
            goal.rank = self.ranking[i]
            goal.prob = self.probs[i]

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


def apply_obstacles_network(scene_config, network):
    G = network.copy()
    obs = [tuple(ob) for ob in scene_config.ob]
    for u,v in G.edges(obs):
        G[u][v]['weight'] = 10000
    return G



class FastIntentionRecognition(OnlineGoalRecognition):
    def __init__(self, source, goals):
        super().__init__(source, goals)
        self.segment_scores = [[] for i in range(len(goals))]   # len(goals) * len(segments)
        self.segment_score_sum =  np.zeros(len(goals))

    def _compute_segment_score(self, segment, goal):
        length = len(segment) -1
        slope = self._slope(segment[0],segment[-1],goal)/length
        tmp_func = lambda x: 1 if x>0 else NEG_SLOPE_WEIGHT
        h = tmp_func(slope)
        return length*slope/h
        

    def _goal_rank_score(self, goal_idx):
        score = 0
        goal = self.goals[goal_idx]
        lengths = [len(seg)-1 for seg in self.segments]
        if len(self.segments) != len(self.segment_scores[goal_idx]):
            segment_score = self._compute_segment_score(self.segments[-1], goal)
            self.segment_scores[goal_idx].append(segment_score)
            self.segment_score_sum[goal_idx]  = 0
            for i, score in enumerate(self.segment_scores[goal_idx]):
                recency = sum(lengths[i:])
                self.segment_score_sum[goal_idx] += score/recency
            score = self.segment_score_sum[goal_idx]
        if self.current_segment:
            segment_score = self._compute_segment_score(self.current_segment, goal)
            score += segment_score
        return score

    


