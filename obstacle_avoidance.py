import mujoco
import mujoco.viewer
import numpy as np

model_path = "./scene_with_obstacles.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

def get_end_effector_pos(data):
    link6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6")
    return data.xpos[link6_id].copy()

def get_obstacle_positions(data):
    obstacles = []
    for i in range(1, 4):
        obs_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle{i}")
        obstacles.append(data.xpos[obs_id].copy())
    return obstacles

def check_collision_risk(ee_pos, obstacles, threshold=0.15):
    for obs_pos in obstacles:
        dist = np.linalg.norm(ee_pos - obs_pos)
        if dist < threshold:
            return True, obs_pos, dist
    return False, None, None

def compute_avoidance_vector(ee_pos, obs_pos, dist):
    direction = ee_pos - obs_pos
    direction = direction / (np.linalg.norm(direction) + 1e-6)
    strength = max(0, 1.0 - dist / 0.15)
    return direction * strength * 0.3

def plan_safe_trajectory(current_qpos, target_qpos, data, obstacles, steps=200):
    trajectory = []
    for i in range(steps + 1):
        alpha = i / steps
        interpolated = current_qpos * (1 - alpha) + target_qpos * alpha

        data.qpos[:] = interpolated
        mujoco.mj_forward(model, data)

        ee_pos = get_end_effector_pos(data)
        collision, obs_pos, dist = check_collision_risk(ee_pos, obstacles, threshold=0.12)

        if collision:
            avoidance = compute_avoidance_vector(ee_pos, obs_pos, dist)
            for j in range(3):
                interpolated[j] += avoidance[j] * 0.5

        trajectory.append(interpolated.copy())

    return trajectory

home_pose = np.array([0, 1.57, -1.3485, 0, 0, 0, 0, 0])
target1 = np.array([0.8, 1.2, -1.0, 0.3, 0.5, 0, 0, 0])
target2 = np.array([-0.6, 1.8, -1.5, -0.4, -0.3, 0.5, 0, 0])
target3 = np.array([0.4, 1.0, -0.8, 0.2, 0.8, -0.3, 0, 0])

waypoints = [home_pose, target1, target2, target3, home_pose]

data.qpos[:] = home_pose
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    current_waypoint = 0
    trajectory = []
    traj_index = 0

    while viewer.is_running():
        if len(trajectory) == 0 or traj_index >= len(trajectory):
            current_waypoint = (current_waypoint + 1) % len(waypoints)
            next_waypoint = waypoints[current_waypoint]

            obstacles = get_obstacle_positions(data)
            trajectory = plan_safe_trajectory(data.qpos[:].copy(), next_waypoint, data, obstacles)
            traj_index = 0

        target_qpos = trajectory[traj_index]
        data.ctrl[:] = target_qpos[:7]

        mujoco.mj_step(model, data)
        viewer.sync()

        traj_index += 1
        step += 1
