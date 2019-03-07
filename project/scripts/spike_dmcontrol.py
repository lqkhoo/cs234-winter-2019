from lib.dm_control import suite # This hack is so that we can define and use our own envs
from dm_control import viewer
import numpy as np

env = suite.load(domain_name="humanoid_CMU", task_name="stand")
action_spec = env.action_spec()

# Define a uniform random policy.
def random_policy(time_step):
  del time_step  # Unused.
  return np.random.uniform(low=action_spec.minimum,
                           high=action_spec.maximum,
                           size=action_spec.shape)

# Launch the viewer application.
viewer.launch(env, policy=None, width=1024, height=768)
