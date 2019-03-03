from mujoco_py import load_model_from_path, MjSim, functions
import mujoco_py.generated.const as const
import math
import os
from src.models import PointRoller
from src.viewer import Viewer

m = PointRoller(xml_filepath='../assets/pointroller.xml')
sim = MjSim(m.mjmodel)
viewer = Viewer(sim)

t = 0
while True:

    viewer.render()

    id_body, _ = m.name2id['torso']
    # print("body COM pos: {0}".format(sim.data.xipos[id_body]))

    id_motorx, _ = m.name2id['motorx']
    id_motory, _ = m.name2id['motory']

    # Generally the fields in data we are interested in are:
    # xpos (or xquat) of all bodies. This is our raw observation
    # ctrl (our u)
    # The loss function e.g. the Frobenius norm of (xpos_target - xpos_geom)

    # xmag = math.cos(t / 2/math.pi)
    # ymag = math.sin(t / 2/math.pi)
    # sim.data.ctrl[id_motorx] = xmag
    # sim.data.ctrl[id_motory] = ymag

    print(m.name2id)

    # print("\n")

    t += 1
    sim.step()
    if t > 100 and os.getenv('TESTING') is not None:
        break