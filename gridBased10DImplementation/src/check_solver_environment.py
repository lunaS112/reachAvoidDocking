"""
环境检查脚本：测试ODP或hj_reachability是否可用
"""

import sys
import os

print("=" * 70)
print("SOLVER ENVIRONMENT CHECK FOR 10D QUADROTOR")
print("=" * 70)

# Test 1: Check hj_reachability (JAX)
print("\n[1/3] Checking hj_reachability (JAX)...")
try:
    import jax
    from hj_reachability import dynamics, sets, utils
    print("✓ hj_reachability available")
    print(f"  JAX version: {jax.__version__}")
    HJ_AVAILABLE = True
except ImportError as e:
    print(f"✗ hj_reachability not available: {e}")
    HJ_AVAILABLE = False

# Test 2: Check ODP/HeteroCL
print("\n[2/3] Checking ODP (optimized_dp)...")
try:
    # Try to import heterocl
    import heterocl as hcl
    print("✓ HeteroCL available")
    ODP_AVAILABLE = True
except ImportError:
    print("✗ HeteroCL not available")
    ODP_AVAILABLE = False
except Exception as e:
    print(f"✗ HeteroCL error: {e}")
    ODP_AVAILABLE = False

# Test 3: Small scale test with hj_reachability
if HJ_AVAILABLE:
    print("\n[3/3] Running small-scale hj_reachability test...")
    try:
        import jax.numpy as jnp

        # Simple 2D test
        class Simple2D(dynamics.ControlAndDisturbanceAffineDynamics):
            def __init__(self):
                super().__init__(
                    control_mode="min", disturbance_mode="max",
                    control_space=sets.Box(jnp.array([-1.0, -1.0]), jnp.array([1.0, 1.0])),
                    disturbance_space=sets.Box(jnp.array([-0.1, -0.1]), jnp.array([0.1, 0.1]))
                )

            def open_loop_dynamics(self, state, time=None):
                return jnp.array([-state[1], state[0]])

            def control_jacobian(self, state, time=None):
                return jnp.eye(2)

            def disturbance_jacobian(self, state, time=None):
                return jnp.eye(2)

        sys = Simple2D()
        print("✓ Can instantiate dynamics class")

        # Test evaluation
        state = jnp.array([0.5, 0.3])
        dsdt = sys.open_loop_dynamics(state)
        print(f"✓ Can compute dynamics: dsdt = {dsdt}")
        print("✓ hj_reachability is functional")
        HJ_TEST_PASS = True
    except Exception as e:
        print(f"✗ hj_reachability test failed: {e}")
        HJ_TEST_PASS = False
else:
    HJ_TEST_PASS = False

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"ODP (HeteroCL):        {'✓ Available' if ODP_AVAILABLE else '✗ NOT available'}")
print(f"hj_reachability (JAX): {'✓ Available & functional' if HJ_TEST_PASS else '✗ NOT available'}")

if ODP_AVAILABLE:
    print("\n✓ Will use ODP for HJ solving")
    print("  If ODP fails during solve, will fallback to hj_reachability")
    SOLVER_CHOICE = "ODP"
elif HJ_TEST_PASS:
    print("\n✓ Will use hj_reachability (JAX) for HJ solving")
    SOLVER_CHOICE = "JAX"
else:
    print("\n✗ No solver available! Cannot proceed with Phase 1")
    print("  Please install hj_reachability or HeteroCL")
    sys.exit(1)

print("\n" + "=" * 70)
print(f"SELECTED SOLVER: {SOLVER_CHOICE}")
print("=" * 70)
