#!/usr/bin/env python3
"""
Generate grid-based HJ ground truth for 6D docking at tMax=17.0s.
Uses ComboController to solve the 4D+2D decomposed system.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add gridBased6D to path
grid_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(grid_dir))

from ComboControl import ComboController

def main():
    print("="*60)
    print("Grid-Based HJ Ground Truth Generator (tMax=17.0s)")
    print("="*60)

    # Output directory
    output_dir = grid_dir / "outputs"
    output_dir.mkdir(exist_ok=True)

    cache_dir = output_dir / "grid_cache"
    cache_dir.mkdir(exist_ok=True)

    print(f"\nOutput directory: {output_dir}")
    print(f"Cache directory: {cache_dir}")

    # Create ComboController with tMax=17.0s
    print("\n" + "="*60)
    print("Initializing ComboController (final_time=-17.0)")
    print("="*60)

    try:
        combo = ComboController(
            mc=200.0,
            orbit_alt=400,
            post_hw_x=0.6,
            post_length=0.2,
            w_t=6, h_t=3,
            w_c=1.0, h_c=1.0,
            eps_p=0.1, eps_v=0.1,
            eps_theta=0.04, eps_omega=0.05,
            u_bar_4D=20.0, u_bar_2D=1.5,
            d_bar_4D=0.0, d_bar_2D=0.0,
            final_time=-17.0,  # 17 seconds backward time
            dt=0.5,  # Match the rollout dt
            grid_resolution_4D=(91, 101, 21, 21),
            grid_resolution_2D=(361, 141),
            cache_dir=str(cache_dir),
            filter_mode=2
        )

        print("✓ ComboController initialized successfully")
        print(f"  - 4D Value function shape: {combo.values_4D.shape}")
        print(f"  - 2D Value function shape: {combo.values_2D.shape}")
        print(f"  - Time steps: {len(combo.times)}")
        print(f"  - Time range: {combo.times[0]:.2f} to {combo.times[-1]:.2f}s")

    except Exception as e:
        print(f"\n✗ Error during ComboController initialization:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "="*60)
    print("Grid baseline generation complete!")
    print("="*60)
    print(f"\nValue functions cached at: {cache_dir}")
    print("Ready for comparison with neural network controllers.")

    return 0

if __name__ == "__main__":
    sys.exit(main())
