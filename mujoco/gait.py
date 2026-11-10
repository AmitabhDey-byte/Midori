"""Joint layout and crawl gait calculations for the Midori quadruped."""

import math

DATASET_SAMPLE_PERIOD_SECONDS = 0.02
GAIT_FREQUENCY_HZ = 0.85
STRIDE_METRES = 0.040
TURN_STRIDE_METRES = 0.040
SWING_FRACTION = 0.25
FOOT_LIFT_METRES = 0.030
STANCE_PRELOAD_METRES = 0.002
SUPPORT_SHIFT_FRACTION = 0.70
STANCE_SEARCH_RATE_MPS = 0.040
STANCE_SEARCH_MAX_METRES = 0.008
STANCE_RELEASE_RATE_MPS = 0.080
GAIT_CONTROLLER_VERSION = "stance_transition_sequential_crawl_v3"
WALK_STANCE_FEMUR_ANGLE = math.radians(-45.0)
WALK_STANCE_TIBIA_ANGLE = math.radians(65.0)
WALK_STANCE_TRANSITION_SECONDS = 1.2
WALK_GAIT_BLEND_SECONDS = 0.35
SIM_SERVO_KP = 10.0
SIM_SERVO_KV = 0.5
SIM_SERVO_TORQUE_LIMIT_NM = 0.45
JOINTS = [
    ("Front left coxa yaw", "front_left_hip_abduction"),
    ("Front left femur pitch", "front_left_hip_pitch"),
    ("Front left tibia pitch", "front_left_knee_pitch"),
    ("Front right coxa yaw", "front_right_hip_abduction"),
    ("Front right femur pitch", "front_right_hip_pitch"),
    ("Front right tibia pitch", "front_right_knee_pitch"),
    ("Rear left coxa yaw", "rear_left_hip_abduction"),
    ("Rear left femur pitch", "rear_left_hip_pitch"),
    ("Rear left tibia pitch", "rear_left_knee_pitch"),
    ("Rear right coxa yaw", "rear_right_hip_abduction"),
    ("Rear right femur pitch", "rear_right_hip_pitch"),
    ("Rear right tibia pitch", "rear_right_knee_pitch"),
]


LEG_INFO = {
    "front_left": (0.032, 0.027, math.pi / 4, 0.00),
    "front_right": (0.032, -0.027, -math.pi / 4, 0.50),
    "rear_left": (-0.032, 0.027, 3 * math.pi / 4, 0.25),
    "rear_right": (-0.032, -0.027, -3 * math.pi / 4, 0.75),
}


COXA_LENGTH = 0.036
FEMUR_VECTOR = (0.030, 0.040)
TIBIA_VECTOR = (0.05975, 0.0988)
FEMUR_LENGTH = math.hypot(*FEMUR_VECTOR)
TIBIA_LENGTH = math.hypot(*TIBIA_VECTOR)
FEMUR_BASE_ANGLE = math.atan2(FEMUR_VECTOR[0], FEMUR_VECTOR[1])
TIBIA_BASE_ANGLE = math.atan2(TIBIA_VECTOR[0], TIBIA_VECTOR[1])
REST_FEMUR_ANGLE = -0.55
REST_TIBIA_ANGLE = 1.10
FOOT_RADIUS = 0.0015


def leg_foot_geometry(femur_angle, tibia_angle):
    """Return radial reach and downward reach for a leg's current pitch angles."""
    total_femur_angle = femur_angle
    total_tibia_angle = femur_angle + tibia_angle
    femur_x = FEMUR_VECTOR[0] * math.cos(total_femur_angle) - FEMUR_VECTOR[1] * math.sin(total_femur_angle)
    femur_down = FEMUR_VECTOR[0] * math.sin(total_femur_angle) + FEMUR_VECTOR[1] * math.cos(total_femur_angle)
    tibia_x = TIBIA_VECTOR[0] * math.cos(total_tibia_angle) - TIBIA_VECTOR[1] * math.sin(total_tibia_angle)
    tibia_down = TIBIA_VECTOR[0] * math.sin(total_tibia_angle) + TIBIA_VECTOR[1] * math.cos(total_tibia_angle)
    return COXA_LENGTH + femur_x + tibia_x, femur_down + tibia_down


def smoothstep01(value):
    """Quintic easing with zero velocity and acceleration at both endpoints."""
    value = max(0.0, min(1.0, value))
    return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)


_, REST_FOOT_DROP = leg_foot_geometry(REST_FEMUR_ANGLE, REST_TIBIA_ANGLE)
ROOT_STAND_HEIGHT = REST_FOOT_DROP + FOOT_RADIUS


def inverse_kinematics(x_from_coxa, y_from_coxa, foot_down):
    """Solve coxa yaw and two pitch angles for a foot point in a leg's frame."""
    coxa_yaw = math.atan2(y_from_coxa, x_from_coxa)
    radial_distance = math.hypot(x_from_coxa, y_from_coxa)
    planar_x = radial_distance - COXA_LENGTH
    distance = math.hypot(planar_x, foot_down)

    minimum_reach = abs(FEMUR_LENGTH - TIBIA_LENGTH) + 1e-5
    maximum_reach = FEMUR_LENGTH + TIBIA_LENGTH - 1e-5
    distance = min(max(distance, minimum_reach), maximum_reach)

    cos_delta = (
        distance * distance - FEMUR_LENGTH * FEMUR_LENGTH - TIBIA_LENGTH * TIBIA_LENGTH
    ) / (2 * FEMUR_LENGTH * TIBIA_LENGTH)


    delta = -math.acos(max(-1.0, min(1.0, cos_delta)))
    target_angle = math.atan2(planar_x, foot_down)
    inner_angle = math.atan2(
        TIBIA_LENGTH * math.sin(delta),
        FEMUR_LENGTH + TIBIA_LENGTH * math.cos(delta),
    )
    femur_angle = FEMUR_BASE_ANGLE - (target_angle - inner_angle)
    tibia_angle = (TIBIA_BASE_ANGLE - FEMUR_BASE_ANGLE) - delta
    return coxa_yaw, femur_angle, tibia_angle


def crawl_support_compensation(cycle, base_pose):
    """Shift the body target toward each three-foot support triangle smoothly."""
    legs = []
    for leg_name, (hip_x, hip_y, heading, phase_offset) in LEG_INFO.items():
        base_yaw = (base_pose or {}).get(f"{leg_name}_hip_abduction", 0.0)
        base_femur = (base_pose or {}).get(f"{leg_name}_hip_pitch", REST_FEMUR_ANGLE)
        base_tibia = (base_pose or {}).get(f"{leg_name}_knee_pitch", REST_TIBIA_ANGLE)
        reach, _ = leg_foot_geometry(base_femur, base_tibia)
        world_heading = heading + base_yaw
        nominal_foot = (
            hip_x + reach * math.cos(world_heading),
            hip_y + reach * math.sin(world_heading),
        )
        legs.append((leg_name, nominal_foot, phase_offset))

    legs.sort(key=lambda item: (item[2] + SWING_FRACTION / 2.0) % 1.0)
    supports = {}
    for leg_name, _, _ in legs:
        points = [point for other_name, point, _ in legs if other_name != leg_name]
        supports[leg_name] = (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )

    centers = [((offset + SWING_FRACTION / 2.0) % 1.0, name) for name, _, offset in legs]
    centers.sort()
    phase = cycle % 1.0
    for index, (left_center, left_leg) in enumerate(centers):
        next_center, next_leg = centers[(index + 1) % len(centers)]
        phase_value = phase
        if index == len(centers) - 1:
            next_center += 1.0
            if phase_value < left_center:
                phase_value += 1.0
        elif phase_value < left_center:
            continue
        if left_center <= phase_value <= next_center:
            blend = smoothstep01((phase_value - left_center) / (next_center - left_center))
            center_x = supports[left_leg][0] + blend * (supports[next_leg][0] - supports[left_leg][0])
            center_y = supports[left_leg][1] + blend * (supports[next_leg][1] - supports[left_leg][1])
            return -SUPPORT_SHIFT_FRACTION * center_x, -SUPPORT_SHIFT_FRACTION * center_y
    return 0.0, 0.0


def crawl_swinging_legs(gait_time):
    """Return the legs currently in swing for the sequential crawl phase."""
    cycle = gait_time * GAIT_FREQUENCY_HZ
    return {
        leg_name
        for leg_name, (_, _, _, phase_offset) in LEG_INFO.items()
        if (cycle + phase_offset) % 1.0 < SWING_FRACTION
    }


def crawl_targets(gait_time, mode, base_pose=None, blend=1.0, stance_corrections=None):
    """Generate crawl targets around a captured manual pose, then solve 3-DoF IK."""
    targets = {}
    if mode == "stop":
        for _, name in JOINTS:
            targets[name] = REST_FEMUR_ANGLE if name.endswith("hip_pitch") else (
                REST_TIBIA_ANGLE if name.endswith("knee_pitch") else 0.0
            )
        return targets

    blend = max(0.0, min(1.0, blend))

    direction = -1.0 if mode == "backward" else 1.0


    turn_sign = -1.0 if mode == "left" else 1.0
    stride_metres = TURN_STRIDE_METRES if mode in ("left", "right") else STRIDE_METRES


    cycle = gait_time * GAIT_FREQUENCY_HZ
    support_comp_x, support_comp_y = crawl_support_compensation(cycle, base_pose)

    for leg_name, (hip_x, hip_y, heading, phase_offset) in LEG_INFO.items():
        base_yaw = (base_pose or {}).get(f"{leg_name}_hip_abduction", 0.0)
        base_femur = (base_pose or {}).get(f"{leg_name}_hip_pitch", REST_FEMUR_ANGLE)
        base_tibia = (base_pose or {}).get(f"{leg_name}_knee_pitch", REST_TIBIA_ANGLE)
        base_reach, base_foot_down = leg_foot_geometry(base_femur, base_tibia)
        base_heading = heading + base_yaw
        phase = (cycle + phase_offset) % 1.0
        if phase < SWING_FRACTION:
            swing_progress = smoothstep01(phase / SWING_FRACTION)


            stride_offset = (-0.5 + swing_progress) * stride_metres
            foot_lift = FOOT_LIFT_METRES * math.sin(math.pi * swing_progress)
        else:
            stance_progress = smoothstep01(
                (phase - SWING_FRACTION) / (1.0 - SWING_FRACTION)
            )


            stride_offset = (0.5 - stance_progress) * stride_metres
            foot_lift = 0.0

        if mode in ("forward", "backward"):
            travel_x = direction
            travel_y = 0.0
        else:

            travel_x = turn_sign * math.sin(base_heading)
            travel_y = -turn_sign * math.cos(base_heading)

        foot_world_x = (
            hip_x + base_reach * math.cos(base_heading)
            + stride_offset * travel_x + support_comp_x
        )
        foot_world_y = (
            hip_y + base_reach * math.sin(base_heading)
            + stride_offset * travel_y + support_comp_y
        )
        relative_x = foot_world_x - hip_x
        relative_y = foot_world_y - hip_y
        local_x = math.cos(heading) * relative_x + math.sin(heading) * relative_y
        local_y = -math.sin(heading) * relative_x + math.cos(heading) * relative_y
        stance_correction = (stance_corrections or {}).get(leg_name, 0.0)
        if phase >= SWING_FRACTION and stance_progress > 0.85:
            release_progress = smoothstep01((stance_progress - 0.85) / 0.15)
            stance_correction *= 1.0 - release_progress
        foot_down = (
            base_foot_down + STANCE_PRELOAD_METRES + stance_correction - foot_lift
        )
        yaw, femur, tibia = inverse_kinematics(local_x, local_y, foot_down)

        base_yaw = (base_pose or {}).get(f"{leg_name}_hip_abduction", 0.0)
        base_femur = (base_pose or {}).get(f"{leg_name}_hip_pitch", REST_FEMUR_ANGLE)
        base_tibia = (base_pose or {}).get(f"{leg_name}_knee_pitch", REST_TIBIA_ANGLE)
        targets[f"{leg_name}_hip_abduction"] = base_yaw + blend * (yaw - base_yaw)
        targets[f"{leg_name}_hip_pitch"] = base_femur + blend * (femur - base_femur)
        targets[f"{leg_name}_knee_pitch"] = base_tibia + blend * (tibia - base_tibia)

    return targets


def foot_contact_flags(data, foot_geom_ids, floor_geom_id):
    """Return whether each named foot geom currently contacts the floor geom."""
    contacts = {leg_name: False for leg_name in foot_geom_ids}
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        for leg_name, foot_geom_id in foot_geom_ids.items():
            if {contact.geom1, contact.geom2} == {foot_geom_id, floor_geom_id}:
                contacts[leg_name] = True
    return contacts


