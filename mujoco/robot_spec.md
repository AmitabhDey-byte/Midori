# Midori — Robot Specification

## Target design

- Printed Bots quadruped spider frame kit with four legs.
- Intended control layout: 12 servos, three per leg (coxa yaw, femur pitch, tibia/knee pitch).
- The kit is sold without servos; confirm the printed bracket version against the selected servo and horn size before hardware assembly.
- The supplied STL parts are stored in `mujoco/assets/` and used as articulated visual meshes in `hello.xml`.
- Mesh dimensions below are file bounds, not calibrated joint-center distances.

## 1. Body and printed parts

| Part | Approximate STL bounds (mm) | Current MuJoCo use |
|---|---:|---|
| `body-d.stl` | 83.5 × 75.8 × 13.9 | Lower body frame visual |
| `body-u.stl` | 85.5 × 76.5 × 8.0 | Upper body frame visual |
| `body-m.stl` | 71.6 × 84.9 × 8.9 | Middle plate visual |
| `coxa-f.stl`, `coxa-b.stl` | 38.1 × 39.3 × 20.2 | Shoulder bracket visuals |
| `femur.stl` | 49.5 × 40.2 × 12.0 | Articulated upper-link visual |
| `tibia-f.stl`, `tibia-b.stl` | 100.4 × 25.0 × 16.6 | Articulated pointed claw visuals |
| `hinge.stl` | 16.1 × 12.0 × 6.8 | Printed joint bracket visuals |

The CAD STL files use a shared assembly/print coordinate frame. `hello.xml` maps those shapes into the robot's X-forward, Y-left, Z-up convention. Mesh attachment offsets and some part-to-leg assignments are visual estimates from the shared assembly image; check them against the physical assembly before treating the scene as a digital twin.

## 2. Current simulated geometry and control

- Four legs with 3 revolute joints per leg: coxa yaw, femur pitch, tibia pitch.
- Position actuators exist for all 12 joints.
- The supplied STL meshes are visual-only. Simplified boxes/capsules provide the main collision and mass proxies, and small hidden toe contacts provide foot-ground contact.
- Gravity remains enabled at `-9.81 m/s²`; virtual pose stabilization is off by default. If explicitly enabled, it applies an external chassis spring-damper while stopped and overrides natural falling.
- Current kinematic link projections are 36 mm coxa, 50 mm femur, and about 115 mm from knee pivot to STL claw tip. The tibia endpoint was updated from measured STL vertex placement; joint centers and segment masses remain estimates.
- Hidden toe contact spheres now sit at the visible claw tips. Tibia collision capsules are narrower than the toe spheres so the printed claws contact the floor first.
- Simulation advances with a wall-clock accumulator at the 2 ms model timestep. Physics steps are batched between viewer refreshes for approximately real-time motion.
- Virtual position actuators are tuned to 0.45 Nm for clearer simulation joint tracking. This setting is not SG90 calibrated and must not be copied to hardware without measuring the installed servos.
- Femur pitch simulator range is currently -86° to +60° for all four legs. Confirm the physical servo/bracket hard stop before using this range on hardware.
- A low four-beat crawl swings one foot at a time. Direction controls operate while held and stop on release.
- Gravity is enabled at `-9.81 m/s²` along world Z.

## 3. Still unknown or requiring measurement

- Servo make/model, horn type, output shaft offsets, and mounting clearances.
- Exact joint centers and pivot-to-pivot distances.
- Servo neutral angles, rotation signs, hard stops, and safe operating ranges.
- Printed frame mass, total assembled mass, and center of mass.
- Correct physical collision geometry, surface friction, and claw contact point.
- Final stance footprint, body ground clearance, and loaded deflection.

Measure the physical robot before transferring joint commands from the simulator to the ESP32. STL visual geometry improves shape and proportions but does not establish safe servo calibration or physical dynamics.
