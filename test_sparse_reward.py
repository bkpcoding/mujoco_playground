#!/usr/bin/env python3
"""Test script for the sparse reward T1 environment."""

try:
    from mujoco_playground._src.locomotion import load
    print("✓ Import successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

def test_sparse_environment():
    print("Testing T1SparseJoystickFlatTerrain environment...")
    
    try:
        # Load the sparse environment
        env = load('T1SparseJoystickFlatTerrain')
        print("✓ Environment loaded successfully!")
        
        # Print active reward components (non-zero weights)
        print("\nActive sparse reward components:")
        for k, v in env._config.reward_config.scales.items():
            if v != 0.0:
                print(f"  {k}: {v}")
        
        # Print sparse-specific configuration
        print("\nSparse-specific configuration:")
        print(f"  tracking_threshold: {env._config.reward_config.tracking_threshold}")
        print(f"  tracking_sigma: {env._config.reward_config.tracking_sigma}")
        print(f"  milestone_distance: {env._config.reward_config.milestone_distance}")
        print(f"  stability_duration: {env._config.reward_config.stability_duration}")
        
        # Test environment reset
        import jax
        rng = jax.random.PRNGKey(42)
        state = env.reset(rng)
        print(f"✓ Environment reset successful")
        print(f"  State type: {type(state)}")
        if hasattr(state, 'obs'):
            if isinstance(state.obs, dict):
                print(f"  Obs keys: {list(state.obs.keys())}")
                for k, v in state.obs.items():
                    print(f"    {k}: shape {v.shape}")
            else:
                print(f"  Obs shape: {state.obs.shape}")
        else:
            print(f"  State keys: {list(state.keys()) if isinstance(state, dict) else 'Not a dict'}")
        
        # Test a single step
        rng, action_key = jax.random.split(rng)
        action = env.sample_actions(state.info if hasattr(state, 'info') else state.get('info', {}), action_key)
        next_state = env.step(state, action)
        print(f"✓ Environment step successful")
        
        # Print reward breakdown
        print("\nReward breakdown from first step:")
        reward_dict = next_state.reward if hasattr(next_state, 'reward') else next_state.get('reward', {})
        if isinstance(reward_dict, dict):
            for k, v in reward_dict.items():
                print(f"  {k}: {float(v):.4f}")
            
            if 'reward' in reward_dict:
                print(f"\nTotal reward: {float(reward_dict['reward']):.4f}")
        else:
            print(f"  Total reward: {float(reward_dict):.4f}")
        
        print("\n✓ All tests passed! Sparse reward environment is working correctly.")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sparse_environment()