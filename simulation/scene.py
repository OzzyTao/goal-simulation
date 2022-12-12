class SceneConfig:
    def __init__(self,width, height,ob_points, source_loc, goal_loc):
        self.ob = ob_points
        self.source_loc = source_loc
        self.goal_loc = goal_loc
        self.width = width
        self.height = height

class RandomScene:
    def __init__(self, random, width, height, ob_patch_num) -> None:
        self.random = random 
        self.width = width
        self.height = height
        self.ob_patch_num = ob_patch_num
        self.scene = self._create_scene()
        self.destinations = []

    def _create_scene(self):
        obs = []
        to_visit_stack = []
        while len(obs)<self.ob_patch_num:
            if len(to_visit_stack)==0:
                to_visit_stack.append([self.random.randrange(self.width),self.random.randrange(self.height)])
            current_idx = self.random.randrange(len(to_visit_stack))
            current_point = to_visit_stack.pop(current_idx)
            obs.append(current_point)
            for dx,dy in [[0,-1],[0,1],[-1,0],[1,0]]:
                if (0 <= dx+current_point[0] < self.width) and (0 <= dy+current_point[1] < self.height):
                    tmp_loc = [dx+current_point[0],dy+current_point[1]]
                    if tmp_loc not in to_visit_stack and tmp_loc not in obs:
                        to_visit_stack.append(tmp_loc)
        source = []
        while len(source)==0:
            tmp = [self.random.randrange(self.width),self.random.randrange(self.height)]
            if tmp not in obs:
                source = tmp
        target = []
        while len(target)==0:
            tmp = [self.random.randrange(self.width),self.random.randrange(self.height)]
            if tmp not in obs and tmp != source:
                target = tmp
        return SceneConfig(self.width,self.height,obs,source,target)

    def add_candidate_destinations(self, number):
        scene_config = self.scene
        prediction_goals = []
        avoid_poses = [scene_config.source_loc,scene_config.goal_loc] + scene_config.ob
        while len(prediction_goals) < number:
            tmp_pos = [self.random.randrange(scene_config.width),self.random.randrange(scene_config.height)]
            if tmp_pos not in avoid_poses:
                prediction_goals.append(tmp_pos)
                avoid_poses.append(tmp_pos)
        self.destinations = self.destinations + prediction_goals


class FixedScene:
    def __init__(self,random, width, height, ob_patch_width, ob_patch_height) -> None:
        self.random = random
        self.width = width
        self.height = height
        self.ob_width = ob_patch_width
        self.ob_height = ob_patch_height
        self.scene = self._create_scene()
        self.destinations = []

    def _create_scene(self):
        obs = []
        center = [int(self.width/2.0),int(self.height/2.0)]
        dx = int(self.ob_width/2.0)
        dy = int(self.ob_height/2.0)
        for i in range(-dx, dx):
            for j in range(-dy,dy):
                obs.append([center[0]+i,center[1]+j])
        source = [center[0],1]
        target = []
        return SceneConfig(self.width,self.height,obs,source,target)

    def _get_zone_point(self, zone_number):
        center = [int(self.width/2.0),int(self.height/2.0)]
        dx = int(self.ob_width/2.0)
        dy = int(self.ob_height/2.0)
        xmin = center[0]-dx 
        xmax = center[0]+dx 
        ymin = center[1]-dy 
        ymax = center[1]+dy
        if zone_number == 1:
            zone_xmin = 0
            zone_xmax = xmin
            zone_ymin = ymax 
            zone_ymax = self.height 
        elif zone_number == 2:
            zone_xmin = xmin
            zone_xmax = xmax 
            zone_ymin = ymax 
            zone_ymax = self.height 
        elif zone_number == 3:
            zone_xmin = xmax 
            zone_xmax = self.width
            zone_ymin = ymax 
            zone_ymax = self.height 
        elif zone_number == 4:
            zone_xmin = 0
            zone_xmax = xmin
            zone_ymin = ymin
            zone_ymax = ymax
        elif zone_number == 6:
            zone_xmin = xmax
            zone_xmax = self.width
            zone_ymin = ymin
            zone_ymax = ymax
        elif zone_number == 7:
            zone_xmin = 0
            zone_xmax = xmin
            zone_ymin = 0
            zone_ymax = ymin 
        elif zone_number == 8:
            zone_xmin = xmin
            zone_xmax = xmax
            zone_ymin = 0
            zone_ymax = ymin 
        elif zone_number == 9:
            zone_xmin = xmax
            zone_xmax = self.width
            zone_ymin = 0
            zone_ymax = ymin 
        else:
            zone_xmin = 0
            zone_xmax = self.width
            zone_ymin = 0
            zone_ymax = self.height
        return [self.random.randrange(zone_xmin,zone_xmax),self.random.randrange(zone_ymin,zone_ymax)]

    def set_destination(self, zone_number):
        while len(self.scene.goal_loc)==0:
            tmp_point = self._get_zone_point(zone_number=zone_number)
            if (tmp_point != self.scene.source_loc) and (tmp_point not in self.scene.ob):
                self.scene.goal_loc = tmp_point
    
    def set_candidate_desinations(self, number, zone_number=0):
        occupied = self.scene.ob + [self.scene.source_loc, self.scene.goal_loc] + self.destinations
        i = 0
        while i<number:
            tmp_point = self._get_zone_point(zone_number=zone_number)
            if tmp_point not in occupied:
                self.destinations.append(tmp_point)
                occupied.append(tmp_point)
                i += 1
        

