from mujoco_py import load_model_from_path, MjSim, MjViewer, functions
import mujoco_py.generated.const as const
import math
import os

MODEL_PATH = '../assets/pointroller.xml'
model = load_model_from_path(MODEL_PATH)

sim = MjSim(model)
viewer = MjViewer(sim)

t = 0
while True:

    viewer.render()

    id_body = functions.mj_name2id(sim.model, const.OBJ_BODY, 'torso')
    print("body COM pos: {0}".format(sim.data.xipos[id_body]))

    id_motorx = functions.mj_name2id(sim.model, const.OBJ_ACTUATOR, 'motorx')
    id_motory = functions.mj_name2id(sim.model, const.OBJ_ACTUATOR, 'motory')
    xmag = math.cos(t / 2/math.pi)
    ymag = math.sin(t / 2/math.pi)
    sim.data.ctrl[id_motorx] = xmag
    sim.data.ctrl[id_motory] = ymag

    print("\n")


    t += 1
    sim.step()
    if t > 100 and os.getenv('TESTING') is not None:
        break