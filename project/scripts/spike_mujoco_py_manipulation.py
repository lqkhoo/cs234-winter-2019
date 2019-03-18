from mujoco_py import load_model_from_path, MjSim, functions
import mujoco_py.generated.const as const
import multiprocessing
import math
import os
from src.viewer import Viewer
import numpy as np

if __name__ == '__main__':
    multiprocessing.freeze_support()
    MODEL_PATH = '../assets/examples/gym/ant.xml'
    model = load_model_from_path(MODEL_PATH)

    sim = MjSim(model)
    viewer = Viewer(sim) # We are going to be using our custom viewer for this one

    t = 0
    while True:

        viewer.render()

        # id_body = functions.mj_name2id(sim.model, const.OBJ_BODY, 'torso')

        hovered_geom_id, hovered_world_coords = viewer.cursor_raycast()
        print(hovered_geom_id)
        print(sim.data.geom_xpos[hovered_geom_id])

        t += 1
        sim.step()
        if t > 100 and os.getenv('TESTING') is not None:
            break
    
    