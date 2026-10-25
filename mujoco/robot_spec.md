# Midori — Robot Specification

## Target design

- Printed Bots Quadruped Spider Robot, 3D-printed frame kit.
- The linked kit is sold without servos. Midori's intended assembled configuration is 12 servos: 3 per leg.
- The seller describes two frame variants for original-size and larger-clone servo horns. Confirm which version matches the printed parts before finalizing servo mounts.
- The model below is a spider-style approximation, not a dimensionally accurate twin. The listing does not provide CAD dimensions or joint limits.

## 1. Main chassis

- Length: UNKNOWN (current MuJoCo placeholder: 130 mm)
- Width: UNKNOWN (current MuJoCo placeholder: 90 mm)
- Height: UNKNOWN

## 2. Legs

- Number of legs: 4
- Intended servos: 12 total, 3 per leg; servo model not confirmed
- Leg joints: coxa yaw, femur pitch, tibia/knee pitch
- Coxa length: UNKNOWN (current MuJoCo placeholder: 30 mm)
- Femur length: UNKNOWN (current MuJoCo placeholder: approximately 57 mm)
- Tibia length: UNKNOWN (current MuJoCo placeholder: approximately 71 mm)
- Total standing height: UNKNOWN

## 3. Weight

- 3D-printed frame weight: UNKNOWN
- Total assembled weight: UNKNOWN

## 4. Measurements to verify

- [ ] Chassis length, width, and thickness
- [ ] Coxa, femur, and tibia pivot-to-pivot lengths
- [ ] Servo model and kit version (original-size or larger-clone horn)
- [ ] Joint axes, servo horn zero positions, and safe angular limits
- [ ] Total assembled weight and center of mass
