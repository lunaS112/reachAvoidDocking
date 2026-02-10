# 13D Spacecraft Docking Game

An interactive game to control a chaser spacecraft using 13D dynamics and attempt to dock with a target satellite.

## Overview

This game implements the full 13D spacecraft docking dynamics:
- **State (13D)**: `[x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz]`
  - Position in LVLH frame
  - Velocity in LVLH frame  
  - Quaternion orientation (scalar-first)
  - Angular velocity in body frame

- **Control (6D)**: `[Fx, Fy, Fz, tx, ty, tz]`
  - Body-frame forces (N)
  - Body-frame torques (N·m)

The dynamics are **control-affine**, meaning translation and rotation are decoupled - you can verify this by using only translation or only rotation controls.

## Controls

### Translation (Body Frame)
| Key | Action |
|-----|--------|
| W | Forward (+X body) |
| S | Backward (-X body) |
| A | Left (-Y body) |
| D | Right (+Y body) |
| Q | Down (-Z body) |
| E | Up (+Z body) |

### Rotation (Body Frame)
| Key | Action |
|-----|--------|
| ↑ | Pitch up |
| ↓ | Pitch down |
| ← | Yaw left |
| → | Yaw right |
| Z | Roll left |
| C | Roll right |

### Other
| Key | Action |
|-----|--------|
| R | Reset game |
| ESC | Quit |
| Right-drag | Orbit camera |
| Scroll | Zoom in/out |

## Goal

Dock the chaser spacecraft (blue cube) at the origin (yellow marker) with:
- Position error < 0.5 m
- Velocity < 0.1 m/s
- Angular velocity < 0.1 rad/s

## Installation

```bash
# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Game

```bash
python game.py
```

## Physics Details

The game uses:
- **Hill-Clohessy-Wiltshire (HCW)** equations for relative orbital motion
- **Quaternion-based attitude dynamics** with Euler's rotational equations
- **RK4 integration** for numerical stability
- Parameters based on a 200kg spacecraft at 400km orbit altitude

## Verifying Control-Affine Dynamics

To verify that translation and rotation are truly separate:

1. **Translation test**: Use only WASD/Q/E keys - the spacecraft should translate without rotating
2. **Rotation test**: Use only arrow/Z/C keys - the spacecraft should rotate without translating (except for orbital coupling from HCW)

Note: The HCW equations introduce coupling between position and velocity, so the spacecraft will naturally drift in orbit even without control input.
