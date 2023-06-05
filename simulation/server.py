import model
import mesa


def agent_portrayal(agent):
    if isinstance(agent, model.Obstacle) :
        return {
        "Shape": "rect",
        "Color": "gray",
        "Filled": "true",
        "Layer": 0,
        "w": 1,
        "h": 1
        }
    elif isinstance(agent, model.Goal):
        portrayal = {
            "Shape": "circle",
            "Color": "green",
            "Filled": "false",
            "Layer": 1,
            "r": 0.6,
            "text": agent.unique_id,
            "text_color": "black"
        }
        if agent.rank == 0:
            portrayal["r"] = 1
            portrayal["Color"] = "yellow"
        return portrayal
    else:
        return {
        "Shape": "circle",
        "Color": "red",
        "Filled": "true",
        "Layer": 2,
        "r": 0.5,
    }

grid = mesa.visualization.CanvasGrid(agent_portrayal,20,40,500,1000)
server = mesa.visualization.ModularServer(
    model.PathFindingModel,[grid],"Robot demo", {'width':20,'height':40,'obs_num':40,"goal_zones":'2,2,2,2',"path_planning_alg":1}
)
