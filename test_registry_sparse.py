#!/usr/bin/env python3
"""Test script to verify sparse environments are available through the main registry."""

def test_registry_access():
    try:
        from mujoco_playground import registry
        print("✓ Registry import successful")
        
        # Test loading sparse environment through registry
        env_name = "T1SparseJoystickFlatTerrain"
        print(f"\nTesting registry.load('{env_name}')")
        
        # Load default config first
        config = registry.get_default_config(env_name)
        print("✓ Default config loaded")
        
        # Load environment
        env = registry.load(env_name, config=config)
        print("✓ Environment loaded through registry")
        
        # Print available sparse environments
        print(f"\nChecking if sparse environments are in ALL_ENVS:")
        sparse_envs = [name for name in registry.ALL_ENVS if "Sparse" in name]
        if sparse_envs:
            print("✓ Found sparse environments:")
            for sparse_env in sparse_envs:
                print(f"  - {sparse_env}")
        else:
            print("✗ No sparse environments found in registry")
        
        # Test the registry workflow you wanted
        print(f"\nTesting your desired workflow:")
        print(f"from mujoco_playground import registry")
        print(f"raw_env = registry.load('{env_name}', config=train_env_cfg)")
        
        # Show that it works
        train_env_cfg = registry.get_default_config(env_name)
        raw_env = registry.load(env_name, config=train_env_cfg)
        print("✓ Your desired workflow works!")
        
        print(f"\nEnvironment info:")
        print(f"  Type: {type(raw_env)}")
        print(f"  Active sparse rewards: {sum(1 for v in raw_env._config.reward_config.scales.values() if v != 0.0)}")
        
        print(f"\n✓ All registry tests passed!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_registry_access()