# Midori — Robot Specification

## Target design

- Printed Bots Quadruped Spider Robot, 3D-printed frame kit.
- The linked kit is sold without servos. Midori's intended assembled configuration is 12 servos: 3 per leg.
- The seller describes two frame variants for original-size and larger-clone servo horns. Confirm which version matches the printed parts before finalizing servo mounts.
- The current MuJoCo appearance was revised from the supplied perspective CAD screenshot to include the center frame, electronics panel, compact inset servo details, small horn discs, bracketed links, and pointed toe tips (not spherical feet).
- The screenshot is not a dimensioned CAD export. The model remains a spider-style approximation, not a dimensionally accurate twin; the listing/image does not establish overall dimensions, pivot coordinates, masses, or joint limits.

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
- [ ] Overall stance width/length and ground clearance
- [ ] Servo case dimensions and horn orientation at each joint
