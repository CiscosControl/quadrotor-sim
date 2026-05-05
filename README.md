# Quadrotor Formation Control

A simulation of three quadrotor drones flying in formation using LQR control, a Kalman filter, and potential field obstacle avoidance.

---

## Overview

The leader drone tracks a Lissajous figure-8 reference trajectory using LQR feedback + feed-forward control. Two follower drones maintain fixed offsets behind the leader using relative-state LQR. All three drones avoid randomly placed obstacles via repulsive potential fields.

---

## File Structure

| File | Description |
|---|---|
| `sim_spring.py` | Main simulation — runs the ODE, generates plots & animation |
| `controller.py` | LQR gains (leader & follower) and Kalman filter gain |
| `linear_model.py` | Linearised 12-state quadrotor state-space matrices (A, B) |
| `paramaters.py` | Physical parameters (mass, inertia, gravity) |
| `reference.py` | Leader reference trajectory (Lissajous path) |

---

## How It Works

- **Leader** tracks a Lissajous path using LQR + feed-forward (`uff = B⁺(ẋr − Axr)`)
- **Followers** track relative offsets from the leader using separate LQR gains
- **Obstacle avoidance** uses repulsive potential fields applied to x/y acceleration
- **Disturbances** include a simulated wind gust and low-level sensor noise

### State Vector (per drone, 12 states)

```
[x, y, z, vx, vy, vz, φ, θ, ψ, ωx, ωy, ωz]
```

### Inputs (per drone, 4 inputs)

```
[δT (thrust), τx (roll), τy (pitch), τz (yaw)]
```

---

## Output

- **Stability plot** — relative distances between drones over time vs. target offsets
- **State telemetry** — 12-state plots for all three drones
- **Formation error** — x-offset error for each follower
- **Animation** — saved as `drone_formation.gif`

---

## Requirements

```bash
pip install numpy scipy matplotlib
```

## Run

```bash
python sim_spring.py
```

---

## Parameters

| Parameter | Value |
|---|---|
| Mass | 1.5 kg |
| Jx, Jy | 0.02 kg·m² |
| Jz | 0.04 kg·m² |
| Simulation time | 90 s |
| Leader–F1 offset | −2 m (x) |
| Leader–F2 offset | +2 m (x) |
| Obstacles | 15 random |
