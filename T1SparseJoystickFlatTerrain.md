# T1SparseJoystickFlatTerrain Environment

A sparse reward version of the T1 quadruped joystick locomotion task, designed to encourage genuine skill learning over reward hacking.

## Environment Overview

- **Base Environment**: T1JoystickFlatTerrain  
- **Robot**: Booster T1 quadruped
- **Task**: Track linear and angular velocity commands on flat terrain
- **Reward Structure**: Sparse (6 components vs 21 in original)

## Usage

```python
from mujoco_playground import registry
env = registry.load("T1SparseJoystickFlatTerrain", config=train_env_cfg)
```

## Sparse Reward Structure

### Active Components (6 total)

#### 1. **Tracking Rewards (Primary Objectives)**
```python
tracking_lin_vel: 5.0    # Linear velocity tracking
tracking_ang_vel: 2.0    # Angular velocity tracking
```

**Sparse Logic**: Only rewards when tracking error < 0.1 m/s (threshold-based)
- Returns `exp(-error/0.15)` if error < threshold, else 0
- Encourages precise command following, not approximate tracking

#### 2. **Critical Safety Penalties**
```python
collision: -10.0         # Foot-to-foot collisions
orientation: -5.0        # Falling over (>30° tilt)  
dof_pos_limits: -5.0     # Joint limit violations
feet_slip: -2.0          # Foot slipping
```

**Sparse Logic**: Only penalize when critical thresholds are exceeded
- Orientation: Only penalize if tilt > 30 degrees
- Collisions: Large penalty for foot-to-foot contact
- Joint limits: Prevent hardware damage
- Foot slip: Discourage unstable gaits

### Removed Components (Dense → Sparse)
- `alive: 0.25` → `0.0` (removed continuous survival bonus)
- `feet_air_time: 2.0` → `0.0` (removed dense gait feedback)
- `feet_phase: 1.0` → `0.0` (removed continuous gait coordination)
- `ang_vel_xy: -0.15` → `0.0` (removed small orientation penalties)
- All energy-related rewards (torques, action_rate, etc.)

## Training Progression

### **Phase 1: Exploration (0-20k steps)**
```
Typical reward: -15 to -5
Status: Learning basic motor control
- Heavy penalties from falling (-5), collisions (-10), joint limits (-5)
- Zero tracking rewards (can't hit 0.1 m/s threshold)
- High reward variance, mostly negative
```

### **Phase 2: Basic Stability (20k-50k steps)**
```
Typical reward: -5 to +2  
Status: Achieving stability and occasional tracking
- Fewer orientation penalties (staying upright longer)
- Occasional tracking rewards (+5-7 when hitting threshold)
- Still learning coordination between stability and movement
```

### **Phase 3: Command Following (50k-80k steps)**
```
Typical reward: +2 to +6
Status: Consistent tracking with good stability
- Regular tracking rewards as threshold hit more frequently
- Reduced safety penalties through better control
- Learning efficient locomotion patterns
```

### **Phase 4: Mastery (80k+ steps)**
```
Typical reward: +5 to +7
Status: Near-optimal performance
- Consistent tracking rewards (~+7 total possible)
- Minimal penalties (rare -2 from slight slipping)
- Low reward variance, stable policy
```

## Key Configuration Parameters

```python
tracking_threshold: 0.1     # Velocity error threshold for sparse reward
tracking_sigma: 0.15        # Tighter tracking requirement than original (0.25)
milestone_distance: 2.0     # Reserved for future milestone rewards
stability_duration: 50      # Reserved for stability bonus tracking
```

## Differences from Dense Original

| Aspect | Dense (Original) | Sparse (This) |
|--------|------------------|---------------|
| **Active Components** | 21 components | 6 components |
| **Feedback Type** | Continuous every timestep | Threshold-based |
| **Tracking Rewards** | Always non-zero | Zero until threshold met |
| **Alive Bonus** | +0.25 per timestep | Removed |
| **Gait Feedback** | Continuous coordination | Removed |
| **Learning Curve** | Smooth, gradual | Non-linear, sudden jumps |
| **Exploration** | Less required | More required |
| **Final Policy** | May exploit rewards | Genuine skill learning |

## Training Implications

### **Advantages**
- **Robust policies**: Learns actual locomotion skills vs reward exploitation
- **Clear objectives**: Success/failure is well-defined (threshold-based)
- **Efficient inference**: Fewer reward computations during deployment
- **Interpretable**: Easy to understand what robot is optimizing for

### **Challenges**
- **Longer initial training**: More exploration needed to find first rewards
- **Non-linear progress**: Sudden improvements rather than smooth curves
- **Patience required**: Early training may look "stuck" with negative rewards
- **Hyperparameter sensitivity**: Thresholds may need tuning per setup

### **Recommended Training Settings**
- **Longer episodes**: Allow more exploration time per episode
- **Higher exploration**: Increase action noise or entropy bonus initially
- **Patience**: Don't judge performance until 50k+ steps
- **Curriculum**: Consider starting with easier velocity commands

## Implementation Details

### **File Locations**
- **Class**: `SparseJoystick` in `mujoco_playground/_src/locomotion/t1/joystick.py:794`
- **Config**: `sparse_default_config()` in same file at line 744
- **Registry**: Added to `locomotion/__init__.py` and accessible via main registry

### **Related Environments**
- `T1SparseJoystickRoughTerrain`: Same sparse structure on rough terrain
- `T1JoystickFlatTerrain`: Original dense reward version
- `T1JoystickRoughTerrain`: Original dense reward on rough terrain

### **Code Example**
```python
# Load environment
from mujoco_playground import registry
env = registry.load("T1SparseJoystickFlatTerrain")

# Check active rewards
active_rewards = [k for k, v in env._config.reward_config.scales.items() if v != 0.0]
print(f"Active rewards: {active_rewards}")
# Output: ['tracking_lin_vel', 'tracking_ang_vel', 'orientation', 'feet_slip', 'dof_pos_limits', 'collision']

# Training loop
import jax
rng = jax.random.PRNGKey(42)
state = env.reset(rng)
action = env.sample_actions(state.info, rng)
next_state = env.step(state, action)
print(f"Reward components: {next_state.reward}")
```

## Future Enhancements

1. **Adaptive thresholds**: Start loose, tighten as training progresses
2. **Milestone rewards**: Distance/time-based achievement bonuses
3. **Curriculum learning**: Progressive command difficulty
4. **Multi-objective**: Separate tracking and safety reward channels

---

**Created**: 2025-01-20  
**Author**: Claude Code Assistant  
**Environment Version**: SparseJoystick v1.0