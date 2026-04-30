import mujoco
import mujoco.viewer
import numpy as np

# todo 2. 场景三面墙、简单机械臂
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
# todo 3. 给定目标位点，规划三维路径（无碰撞）
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
# todo 1.逆运动学代码实现
def ik_dls(model, data, site_name, target_pos, target_quat=None,
           tol=1e-4, max_iter=100, damping=1e-2, step=0.5):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    nv = model.nv
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))

    for i in range(max_iter):
        mujoco.mj_forward(model, data)
        cur_pos = data.site_xpos[site_id]
        err_pos = target_pos - cur_pos

        if target_quat is not None:
            cur_mat = data.site_xmat[site_id].reshape(3, 3)
            cur_quat = np.zeros(4)
            mujoco.mju_mat2Quat(cur_quat, cur_mat.flatten())
            # 误差四元数: q_err = target * cur^{-1}
            neg_cur = np.zeros(4)
            mujoco.mju_negQuat(neg_cur, cur_quat)
            err_quat = np.zeros(4)
            mujoco.mju_mulQuat(err_quat, target_quat, neg_cur)
            err_rot = np.zeros(3)
            mujoco.mju_quat2Vel(err_rot, err_quat, 1.0)
            err = np.concatenate([err_pos, err_rot])
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
            J = np.vstack([jacp, jacr])
        else:
            err = err_pos
            mujoco.mj_jacSite(model, data, jacp, None, site_id)
            J = jacp

        if np.linalg.norm(err) < tol:
            return data.qpos.copy(), True

        # DLS 求解
        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + damping**2 * np.eye(JJt.shape[0]), err)

        # 用 mj_integratePos 处理自由关节/球关节(正确的 q ⊕ dq)
        mujoco.mj_integratePos(model, data.qpos, dq, step)

    return data.qpos.copy(), False
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
