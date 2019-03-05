from mujoco_py import MjSim
import abc
from idx import Index
from viewer import Viewer

class Task(metaclass=abc.ABCMeta):
    """
    A task consists of our model and a reward function which we need
    to optimize over.
    In here, we also use a task to containerize the simulator.
    """
    def __init__(self, mjsim, render=False):
        self.render = render
        self.sim = mjsim
        self.viewer = Viewer(mjsim)
        self.model = self.sim.model
        self.data = self.model.data
        self.idx = Index(mjmodel=self.model)

    @abc.abstractmethod
    def reset(self):
        """Reset environment to some predefined starting position(s)."""
        pass

    @abc.abstractmethod
    def reward(self, timestep):
        """Return reward at timestep t"""
        return 0


    def expert_output(self):
        """
        Optional.
        Return the output of the expert we want the agent to imitate.
        This is the output of a policy which we assume to be optimal.
        The optimality assumption is necessary* (CS234 Reinforcement Learning)
        """
        pass

    @abc.abstractmethod
    def run(self):
        pass

    # Instead of limiting our observations like in 
    # dm_control, we simply use self.data as observation
