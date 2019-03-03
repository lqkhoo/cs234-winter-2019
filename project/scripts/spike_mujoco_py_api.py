from mujoco_py import load_model_from_path, MjSim, MjViewer, functions
import mujoco_py.generated.const as const

# Load the model as defined in:
# https://github.com/deepmind/dm_control/blob/master/dm_control/suite/humanoid.xml
MODEL_PATH = '../assets/examples/dm/humanoid.xml'
model = load_model_from_path(MODEL_PATH)

def render_callback(sim, viewer):
    pass

sim = MjSim(model, render_callback=render_callback)
viewer = MjViewer(sim)

t = 0
while True:


    # All constants, including enum accessors are listed under:
    # https://github.com/openai/mujoco-py/blob/master/mujoco_py/generated/const.py


    # An MjSim instance has multiple properties.
    # See: https://github.com/openai/mujoco-py/blob/master/mujoco_py/mjsim.pyx
    # For a cython file, properties are listed as cdef expressions


    # sim.get_state() 
    # Returns an MjSimState instance
    # See: https://github.com/openai/mujoco-py/blob/master/mujoco_py/mjsimstate.pyx
    # Has properties: time, qpos, qvel, act, uddstate.
    # These are generally just convenience properties pulled out of PyMjModel and PyMjData,
    # together with simulation statistics like no. of time steps passed.
    # Generally we won't need this unless we need to access time steps.
    state = sim.get_state()


    # sim.model
    # A PyMjModel instance. These properties are generally independent of time
    # See: https://github.com/openai/mujoco-py/blob/c3a4f8b1ff6c20fb271933a24c4dd0beb0728c3c/mujoco_py/generated/wrappers.pxi
    # which wraps: http://www.mujoco.org/book/reference.html#mjModel
    print("n_states: {0}".format(sim.model.nq))
    print("n_DOF: {0}".format(sim.model.nv))
    print("n_ctrl_degrees: {0}".format(sim.model.nu))


    # sim.data 
    # PyMjData instance. These are time-varying properties
    # See: https://github.com/openai/mujoco-py/blob/c3a4f8b1ff6c20fb271933a24c4dd0beb0728c3c/mujoco_py/generated/wrappers.pxi
    # which wraps: http://www.mujoco.org/book/reference.html#mjData
    # Example: To get the velocity of the head:
    id_head = functions.mj_name2id(sim.model, const.OBJ_BODY, 'head')
    id_pelvis = functions.mj_name2id(sim.model, const.OBJ_BODY, 'pelvis')
    print("Head velocity: {0}".format(sim.data.cvel[id_head]))
    print("Pelvis position: {0}".format(sim.data.body_xpos[id_pelvis]))
    print("Pelvis COM position: {0}".format(sim.data.xipos[id_pelvis]))


    # sim.render_contexts
    # List of MjRenderContext. These configure the renderer
    # Generally sim.render_contexts[0] is what we want.
    # See: https://github.com/openai/mujoco-py/blob/master/mujoco_py/mjrendercontext.pyx
    # Has properties:
    # scn   : PyMjvScene    mjNRNDFLAG
    # cam   : PyMjvCamera
    # vopt  : PyMjvOption   mjNVISFLAG
    # pert  : PyMjvPerturb  
    # con   : PyMjvContext
    # Changing properties here changes the rendering.
    # Generally .flags is the one we want. These refer to wrappers listed in 
    # https://github.com/openai/mujoco-py/tree/master/mujoco_py/pxd
    # For MjRenderContext, the one we are interested in is:
    # https://github.com/openai/mujoco-py/blob/6ac6ac203a875ef35b1505827264cadccbfd9f05/mujoco_py/pxd/mjvisualize.pxd
    # Example:
    # Set wireframe rendering to true
    sim.render_contexts[0].scn.flags[const.RND_WIREFRAME] = 1


    # sim.data.ctrl
    # see tutorial_control


    print("\n")

    t += 1
    sim.step()
    viewer.render()
