"""
Utility to load models into mujoco for inspection
"""

from mujoco_py import load_model_from_path, MjSim, MjViewer
import sys

if __name__ == '__main__':
    
    def render_callback(sim, viewer):
        pass

    model_path = sys.argv[1]
    model = load_model_from_path(model_path)
    sim = MjSim(model,
        render_callback=render_callback
    )
    viewer = MjViewer(sim)
    t = 0
    while True:
        t += 1
        sim.step()
        viewer.render()
