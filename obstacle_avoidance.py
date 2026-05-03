import mujoco
import mujoco.viewer
import numpy as np
import threading

model_path = "./scene_with_obstacles.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
_thread_local = threading.local()

def get_ik_data():
    if not hasattr(_thread_local, 'ik_data'):
        _thread_local.ik_data = mujoco.MjData(model)
    return _thread_local.ik_data

# 柱子 XY 位置和半径
PILLARS_XY = np.array([[0.4, 0.4], [-0.4, 0.5], [0.2, 0.6]])
PILLAR_HEIGHTS = np.array([0.8, 0.7, 1.0])   
SAFETY_MARGIN = 0.05                          
PILLAR_R = 0.06 + 0.04 + SAFETY_MARGIN       
WALL_Y_MAX = 1.15 - 0.05
WALL_X_LIM = 1.15 - 0.05

LINK_NAMES = ["base", "link1", "link2", "link3", "link4", "link5", "link6"]
LINK_RADII  = [0.08,   0.04,   0.035,  0.03,   0.025,  0.02,   0.03]
ARM_BODY_IDS = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in LINK_NAMES]
EE_SITE_ID   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

JOINT_LIMITS = np.array([[-2.618,2.618],[-2.094,2.094],[-2.967,2.967],
                          [-1.832,1.832],[-1.22,1.22],[-3.14,3.14]])

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

TARGET_COLORS = np.array([[1.0,0.2,0.2,0.8],[0.2,1.0,0.2,0.8],[0.2,0.2,1.0,0.8],[1.0,1.0,0.2,0.8]], dtype=np.float32)

def draw_targets(viewer, targets, active_idx):
    viewer.user_scn.ngeom = 0
    active = active_idx % len(targets)
    for i, pos in enumerate(targets):
        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.zeros(3), np.zeros(3),
                            np.eye(3).flatten(), TARGET_COLORS[i])
        g.size[:] = 0.07 if i == active else 0.04
        g.pos[:] = pos
        viewer.user_scn.ngeom += 1

with mujoco.viewer.launch_passive(model, data) as viewer:
    current_target = 0
    trajectory = []
    traj_index = 0
    planning = False
    waiting = False
    wait_steps = 0
    sim_step_counter = 0
    TRAJ_ADVANCE_EVERY = 4
    WAIT_DURATION = int(1.0 / model.opt.timestep)  
    _PENDING = object()
    plan_result = [_PENDING]
    start_qpos_for_next = None  

    def run_plan(goal_pos, start_qpos):
        result = rrt_plan(goal_pos, start_qpos)
        if result:
            result = smooth_path(result)
            result = interpolate_path(result)
        plan_result[0] = result if result else []

    while viewer.is_running():
        if waiting:
            if trajectory:
                data.ctrl[:] = trajectory[-1]
            wait_steps += 1
            if wait_steps >= WAIT_DURATION:
                waiting = False
                wait_steps = 0
        elif (not trajectory or traj_index >= len(trajectory)) and not planning:
            goal = cart_targets[current_target % len(cart_targets)]
            print(f"[Target {current_target % len(cart_targets)}] Planning to {goal} ...")
            plan_result[0] = _PENDING
            planning = True
            start_qpos_for_next = data.qpos.copy()  # 锁定规划起点
            threading.Thread(target=run_plan, args=(goal, start_qpos_for_next), daemon=True).start()

        if planning and plan_result[0] is not _PENDING:
            planning = False
            waypoints = plan_result[0]
            if waypoints:
                if np.linalg.norm(waypoints[0] - data.qpos) > 0.05:
                    print(f"  Warning: start position drifted, re-planning...")
                    goal = cart_targets[current_target % len(cart_targets)]
                    plan_result[0] = _PENDING
                    planning = True
                    start_qpos_for_next = data.qpos.copy()
                    threading.Thread(target=run_plan, args=(goal, start_qpos_for_next), daemon=True).start()
                else:
                    trajectory = waypoints
                    traj_index = 0
                    current_target += 1
            else:
                print(f"[Target {current_target % len(cart_targets)}] Skipping.")
                trajectory = []
                traj_index = 0
                current_target += 1

        if trajectory and traj_index < len(trajectory):
            data.ctrl[:] = trajectory[traj_index]

            # 每 TRAJ_ADVANCE_EVERY 步推进轨迹索引
            sim_step_counter += 1
            if sim_step_counter >= TRAJ_ADVANCE_EVERY:
                sim_step_counter = 0
                traj_index += 1

            if traj_index >= len(trajectory):
                waiting = True
                wait_steps = 0
                sim_step_counter = 0
                print(f"[Target {(current_target-1) % len(cart_targets)}] Reached, waiting 1s...")

        mujoco.mj_step(model, data)
        draw_targets(viewer, cart_targets, current_target)
        viewer.sync()
