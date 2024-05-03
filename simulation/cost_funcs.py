"""This module generates the cost functions for the simulation. 
The cost functions are used by the recognition algorithms to rank the goals. 
The cost functions are also used by the simulation to calculate the cost of a trajectory."""


# import osrm
import numpy as np
import networkx as nx
from itertools import combinations

class CostFunction:
    def __init__(self, scene,  cost_func_params):
        self.scene = scene
        self.cost_func_params = cost_func_params

    def __call__(self, start, end):
        return self.cost_func(start, end, **self.cost_func_params)


class EuclideanCost(CostFunction):
    def __init__(self, scene, cost_func_params={}):
        super().__init__(scene, cost_func_params)

    def cost_func(self, start, end):
        return np.linalg.norm(np.array(start)-np.array(end))

class NetworkCost(CostFunction):
    def __init__(self, scene, apply_obstacls=True, cost_func_params={}):
        super().__init__(scene, cost_func_params)
        # self.G = self._create_routing_network()
        self.G = self._create_proximity_network()
        if apply_obstacls:
            self._apply_obstacles(self.G)

    def _create_routing_network(self):
        scene_config = self.scene
        G = nx.Graph()
        nodes = []
        for i in range(scene_config.width):
            for j in range(scene_config.height):
                nodes.append((i,j))
        edges = combinations(nodes, 2)
        weighted_edges = list(map(lambda e: (e[0],e[1],NetworkCost.eu_dist(e[0],e[1])),edges))
        G.add_nodes_from(nodes)
        G.add_weighted_edges_from(weighted_edges)
        return G
    
    def _create_proximity_network(self):
        scene_config = self.scene
        G = nx.Graph()
        nodes = []
        weighted_edges = []
        for i in range(scene_config.width):
            for j in range(scene_config.height):
                nodes.append((i,j))
        edges = combinations(nodes, 2)
        for u,v in edges:
            dist = NetworkCost.eu_dist(u,v)
            if dist < 2:
                weighted_edges.append((u,v,dist))
        G.add_nodes_from(nodes)
        G.add_weighted_edges_from(weighted_edges)
        return G

    def _apply_obstacles(self, G):
        scene_config = self.scene
        obs = [tuple(ob) for ob in scene_config.ob]
        for u,v in G.edges(obs):
            G[u][v]['weight'] = 10000

    def cost_func(self, start, end):
        return nx.astar_path_length(self.G,start,end,heuristic=NetworkCost.eu_dist,weight="weight")

    def eu_dist(start, end):
        return np.linalg.norm(np.array(start)-np.array(end))

# class OSRMCost(CostFunction):
#     def __init__(self, scene, osrm_host, osrm_profile, cost_func_params={}):
#         super().__init__(scene, cost_func_params)
#         self.osrm_config = osrm.DefaultRequestConfig()
#         self.osrm_config.host = osrm_host
#         self.osrm_config.profile = osrm_profile

#     def cost_func(self, start, end):
#         results = osrm.simple_route(start, end, url_config= self.osrm_config)
#         if results:
#             return results[0]['distance']
#         return 1000000000
