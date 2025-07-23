# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Wrappers for MuJoCo Playground environments."""

import functools
from typing import Any, Callable, List, Optional, Sequence, Tuple

from brax.envs.wrappers import training as brax_training
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
import numpy as np
import torch
from mujoco_playground._src import mjx_env
import copy

from jax import tree_util

def sizeof_pytree_mb(pytree) -> float:
    leaves = tree_util.tree_leaves(pytree)
    total_bytes = sum(leaf.size * leaf.dtype.itemsize for leaf in leaves)
    return total_bytes / (1024 * 1024)  # Convert to MB

class Wrapper(mjx_env.MjxEnv):
  """Wraps an environment to allow modular transformations."""

  def __init__(self, env: Any, random_initial_state: bool = False):  # pylint: disable=super-init-not-called
    self.env = env
    self.random_initial_state = random_initial_state

  def reset(self, rng: jax.Array) -> mjx_env.State:
    return self.env.reset(rng)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    return self.env.step(state, action)

  @property
  def observation_size(self) -> mjx_env.ObservationSize:
    return self.env.observation_size

  @property
  def action_size(self) -> int:
    return self.env.action_size

  @property
  def unwrapped(self) -> Any:
    return self.env.unwrapped

  def __getattr__(self, name):
    if name == '__setstate__':
      raise AttributeError(name)
    return getattr(self.env, name)

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self.env.mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self.env.mjx_model

  @property
  def xml_path(self) -> str:
    return self.env.xml_path

  def render(
      self,
      trajectory: List[mjx_env.State],
      height: int = 240,
      width: int = 320,
      camera: Optional[str] = None,
      scene_option: Optional[mujoco.MjvOption] = None,
      modify_scene_fns: Optional[
          Sequence[Callable[[mujoco.MjvScene], None]]
      ] = None,
  ) -> Sequence[np.ndarray]:
    return self.env.render(
        trajectory, height, width, camera, scene_option, modify_scene_fns
    )


def wrap_for_brax_training(
    env: mjx_env.MjxEnv,
    vision: bool = False,
    num_vision_envs: int = 1,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[
        Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]
    ] = None,
    privileged_buffer=None,
    reset_prob: float = 0.5,
    priority_alpha: float = 0.6,
    random_initial_state: bool = False,
) -> Wrapper:
  """Common wrapper pattern for all brax training agents.

  Args:
    env: environment to be wrapped
    vision: whether the environment will be vision based
    num_vision_envs: number of environments the renderer should generate,
      should equal the number of batched envs
    episode_length: length of episode
    action_repeat: how many repeated actions to take per step
    randomization_fn: randomization function that produces a vectorized model
      and in_axes to vmap over

  Returns:
    An environment that is wrapped with Episode and AutoReset wrappers.  If the
    environment did not already have batch dimensions, it is additional Vmap
    wrapped.
  """
  if vision:
    env = MadronaWrapper(env, num_vision_envs, randomization_fn)
  elif randomization_fn is None:
    env = CustomVmapWrapper(env)  # Use our custom wrapper with reset_from_privileged_state
  else:
    env = BraxDomainRandomizationVmapWrapper(env, randomization_fn)
  env = brax_training.EpisodeWrapper(env, episode_length, action_repeat)
  
  # Use prioritized state auto-reset if buffer is provided, otherwise use standard auto-reset
  if privileged_buffer is not None:
    env = PrioritizedStateAutoResetWrapper(
        env, 
        privileged_buffer=privileged_buffer,
        reset_prob=reset_prob,
        priority_alpha=priority_alpha
    )
  else:
    env = BraxAutoResetWrapper(env, random_initial_state)
  
  return env


class BraxAutoResetWrapper(Wrapper):
  """Automatically resets Brax envs that are done."""

  def reset(self, rng: jax.Array) -> mjx_env.State:
    state = self.env.reset(rng)
    state.info['first_state'] = state.data
    state.info['first_obs'] = state.obs
    state.info['raw_obs'] = state.obs
    return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    rng = jax.random.PRNGKey(0)
    # split the rng based on batch size
    rng = jax.random.split(rng, state.done.shape[0])
    if 'steps' in state.info:
        steps = state.info['steps']
        steps = jp.where(state.done, jp.zeros_like(steps), steps)
        state.info.update(steps=steps)
    state = state.replace(done=jp.zeros_like(state.done))
    state = self.env.step(state, action)
    if not self.random_initial_state:
      def where_done(x, y):
        done = state.done
        if done.shape:
          done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
        return jp.where(done, x, y)
      state.info['raw_obs'] = state.obs
      data = jax.tree.map(where_done, state.info['first_state'], state.data)
      obs = jax.tree.map(where_done, state.info['first_obs'], state.obs)
      return state.replace(data=data, obs=obs)
    else:
      # def where_done(x, y):
      #   done = state.done
      #   if done.shape:
      #     done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
      #   return jp.where(done, x, y)
      state.info['raw_obs'] = state.obs
      # data = jax.tree.map(where_done, state.info['first_state'], state.data)
      # obs = jax.tree.map(where_done, state.info['first_obs'], state.obs)
      # Apply normal auto-reset first
      # state = state.replace(data=data, obs=obs)
      state_done_prev = state.done
      # set the first state to a random state
      state = self.env.random_reset(rng, state, state.done)
      # we need to set the done back to the previous state, because that
      # will put the steps backs to 0 using the first command in this function afterwards
      state = state.replace(done=state_done_prev)

      return state

class PrioritizedStateAutoResetWrapper(BraxAutoResetWrapper):
  """Auto-reset wrapper with prioritized privileged state sampling."""
  
  def __init__(
      self,
      env: mjx_env.MjxEnv, 
      privileged_buffer=None,
      reset_prob: float = 0.5,
      priority_alpha: float = 0.6,
      min_buffer_size: int = 100
  ):
    """Initialize prioritized state auto-reset wrapper.
    
    Args:
      env: Environment to wrap
      privileged_buffer: Buffer containing privileged states (PrivilegedStateBuffer or SimpleReplayBuffer)
      reset_prob: Probability of using privileged states vs normal reset (0.5 = 50%)
      priority_alpha: Priority exponent for reward-based sampling (0 = uniform, 1 = proportional)
      min_buffer_size: Minimum buffer size before using privileged resets
    """
    super().__init__(env)
    self.privileged_buffer = privileged_buffer
    self.reset_prob = reset_prob
    self.priority_alpha = priority_alpha
    self.min_buffer_size = min_buffer_size
        

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    """Step with mixed privileged/normal auto-reset logic."""
    # print(f"Stepped through the envs inside privileged brax auto resetter")
    if 'steps' in state.info:
      steps = state.info['steps']
      steps = jp.where(state.done, jp.zeros_like(steps), steps)
      state.info.update(steps=steps)
    state = state.replace(done=jp.zeros_like(state.done))
    state = self.env.step(state, action)
    # print the size of Data in MB
    # print(f"Size of Data: {sizeof_pytree_mb(state.data)} MB")
    def where_done(x, y):
      done = state.done
      if done.shape:
        done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
      return jp.where(done, x, y)
    state.info['raw_obs'] = state.obs
    data = jax.tree.map(where_done, state.info['first_state'], state.data)
    obs = jax.tree.map(where_done, state.info['first_obs'], state.obs)
    # Apply normal auto-reset first
    state = state.replace(data=data, obs=obs)
    
    # Apply privileged reset only to done environments
    if self.privileged_buffer is not None:
      # Sample privileged states from the buffer
      batch_size = 1024 # TODO: make this dynamic
      privileged_state = copy.deepcopy(self.privileged_buffer.sample(batch_size, use_prioritized=True))
      # print(f"Sampled {batch_size} privileged states from buffer")
    else:
      # Fallback to dummy states
      raise ValueError("No privileged buffer provided")
    state = self.env.reset_from_privileged_state(privileged_state, state, state.done)
    return state



class BraxDomainRandomizationVmapWrapper(Wrapper):
  """Brax wrapper for domain randomization."""

  def __init__(
      self,
      env: mjx_env.MjxEnv,
      randomization_fn: Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]],
  ):
    super().__init__(env)
    self._mjx_model_v, self._in_axes = randomization_fn(self.mjx_model)

  def _env_fn(self, mjx_model: mjx.Model) -> mjx_env.MjxEnv:
    env = self.env
    env.unwrapped._mjx_model = mjx_model
    return env

  def reset(self, rng: jax.Array) -> mjx_env.State:
    def reset(mjx_model, rng):
      env = self._env_fn(mjx_model=mjx_model)
      return env.reset(rng)

    state = jax.vmap(reset, in_axes=[self._in_axes, 0])(self._mjx_model_v, rng)
    return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    def step(mjx_model, s, a):
      env = self._env_fn(mjx_model=mjx_model)
      return env.step(s, a)

    res = jax.vmap(step, in_axes=[self._in_axes, 0, 0])(
        self._mjx_model_v, state, action
    )
    return res


class CustomVmapWrapper(Wrapper):
  """Custom Vmap wrapper that includes reset_from_privileged_state method."""

  def __init__(self, env: mjx_env.MjxEnv, batch_size: Optional[int] = None):
    super().__init__(env)
    self.batch_size = batch_size

  def reset(self, rng: jax.Array) -> mjx_env.State:
    if self.batch_size is not None:
      rng = jax.random.split(rng, self.batch_size)
    return jax.vmap(self.env.reset)(rng)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    return jax.vmap(self.env.step)(state, action)

  def _torch_to_jax(self, tensor):
    """Convert PyTorch tensor to JAX array using DLPack."""
    from jax.dlpack import from_dlpack  # pylint: disable=import-outside-toplevel
    return from_dlpack(tensor)

  def random_reset(self, rng: jax.Array, state: mjx_env.State, done_mask: jax.Array) -> mjx_env.State:
    return jax.vmap(self.env.random_reset)(rng, state, done_mask)

  def reset_from_privileged_state(self, obs: Any, state: mjx_env.State, done_mask: jax.Array) -> mjx_env.State:
    """Reset environments from privileged state observations with auto tensor conversion.
    
    Args:
      obs: Dictionary with 'privileged_state' key containing batch of observations.
           Can be JAX array or PyTorch tensor that will be auto-converted.
      state: Current environment state
      done_mask: Boolean array indicating which environments should be reset
      
    Returns:
      Batch of new environment states
    """
    # Handle tensor conversion if needed
    if hasattr(obs, 'numpy'):  # PyTorch tensor
      obs = self._torch_to_jax(obs)
    elif isinstance(obs, dict) and 'privileged_state' in obs:
      privileged_state = obs['privileged_state']
      if hasattr(privileged_state, 'numpy'):  # PyTorch tensor
        obs = {'privileged_state': self._torch_to_jax(privileged_state)}
    
    # Split RNG for batch if needed
    # if self.batch_size is not None:
    #   rng = jax.random.split(rng, self.batch_size)
    
    # Vectorize the single environment reset method
    # def single_reset(obs_single, rng_single):
    #   obs_dict = {'privileged_state': obs_single}
    #   return self.env.reset_from_privileged_state(obs_dict, rng_single)
    
    # Handle both dict and array inputs
    if isinstance(obs, dict):
      privileged_batch = obs['privileged_state']
    else:
      privileged_batch = obs
    
    return jax.vmap(self.env.reset_from_privileged_state, in_axes=(0, 0, 0))(privileged_batch, state, done_mask)


def _identity_vision_randomization_fn(
    mjx_model: mjx.Model, num_worlds: int
) -> Tuple[mjx.Model, mjx.Model]:
  """Tile the necessary fields for the Madrona memory buffer copy."""
  in_axes = jax.tree_util.tree_map(lambda x: None, mjx_model)
  in_axes = in_axes.tree_replace({
      'geom_rgba': 0,
      'geom_matid': 0,
      'geom_size': 0,
      'light_pos': 0,
      'light_dir': 0,
      'light_type': 0,
      'light_castshadow': 0,
      'light_cutoff': 0,
  })
  mjx_model = mjx_model.tree_replace({
      'geom_rgba': jp.repeat(
          jp.expand_dims(mjx_model.geom_rgba, 0), num_worlds, axis=0
      ),
      'geom_matid': jp.repeat(
          jp.expand_dims(jp.repeat(-1, mjx_model.geom_matid.shape[0], 0), 0),
          num_worlds,
          axis=0,
      ),
      'geom_size': jp.repeat(
          jp.expand_dims(mjx_model.geom_size, 0), num_worlds, axis=0
      ),
      'light_pos': jp.repeat(
          jp.expand_dims(mjx_model.light_pos, 0), num_worlds, axis=0
      ),
      'light_dir': jp.repeat(
          jp.expand_dims(mjx_model.light_dir, 0), num_worlds, axis=0
      ),
      'light_type': jp.repeat(
          jp.expand_dims(mjx_model.light_type, 0), num_worlds, axis=0
      ),
      'light_castshadow': jp.repeat(
          jp.expand_dims(mjx_model.light_castshadow, 0), num_worlds, axis=0
      ),
      'light_cutoff': jp.repeat(
          jp.expand_dims(mjx_model.light_cutoff, 0), num_worlds, axis=0
      ),
  })
  return mjx_model, in_axes


def _supplement_vision_randomization_fn(
    mjx_model: mjx.Model,
    randomization_fn: Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]],
    num_worlds: int,
) -> Tuple[mjx.Model, mjx.Model]:
  """Tile the necessary missing fields for the Madrona memory buffer copy."""
  mjx_model, in_axes = randomization_fn(mjx_model)

  required_fields = [
      'geom_rgba',
      'geom_matid',
      'geom_size',
      'light_pos',
      'light_dir',
      'light_type',
      'light_castshadow',
      'light_cutoff',
  ]

  for field in required_fields:
    if getattr(in_axes, field) is None:
      in_axes = in_axes.tree_replace({field: 0})
      val = -1 if field == 'geom_matid' else getattr(mjx_model, field)
      mjx_model = mjx_model.tree_replace({
          field: jp.repeat(jp.expand_dims(val, 0), num_worlds, axis=0),
      })
  return mjx_model, in_axes


class MadronaWrapper:
  """Wraps a MuJoCo Playground to be used in Brax with Madrona."""

  def __init__(
      self,
      env: mjx_env.MjxEnv,
      num_worlds: int,
      randomization_fn: Optional[
          Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]
      ] = None,
  ):
    if not randomization_fn:
      randomization_fn = functools.partial(
          _identity_vision_randomization_fn, num_worlds=num_worlds
      )
    else:
      randomization_fn = functools.partial(
          _supplement_vision_randomization_fn,
          randomization_fn=randomization_fn,
          num_worlds=num_worlds,
      )
    self._env = BraxDomainRandomizationVmapWrapper(env, randomization_fn)
    self.num_worlds = num_worlds

    # For user-made DR functions, ensure that the output model includes the
    # needed in_axes and has the correct shape for madrona initialization.
    required_fields = [
        'geom_rgba',
        'geom_matid',
        'geom_size',
        'light_pos',
        'light_dir',
        'light_type',
        'light_castshadow',
        'light_cutoff',
    ]
    for field in required_fields:
      assert hasattr(self._env._in_axes, field), f'{field} not in in_axes'
      assert (
          getattr(self._env._mjx_model_v, field).shape[0] == num_worlds
      ), f'{field} shape does not match num_worlds'

  def reset(self, rng: jax.Array) -> mjx_env.State:
    """Resets the environment to an initial state."""
    return self._env.reset(rng)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    """Run one timestep of the environment's dynamics."""
    return self._env.step(state, action)

  def __getattr__(self, name):
    """Delegate attribute access to the wrapped instance."""
    return getattr(self._env.unwrapped, name)
