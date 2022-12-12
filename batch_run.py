from model import PathFindingModel
import mesa

params = {'width':30,'height':30,'obs_num':10,"goals_num":2}

results = mesa.batch_run(
    PathFindingModel,
    parameters=params,
    iterations=5,
    max_steps=200,
    display_progress=True
)