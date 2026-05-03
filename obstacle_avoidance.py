import mujoco
import mujoco.viewer
import numpy as np
import threading

model_path = "../scene_with_obstacles.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
_thread_local = threading.local()

def get_ik_data():
    if not hasattr(_thread_local, 'ik_data'):
        _thread_local.ik_data = mujoco.MjData(model)
    return _thread_local.ik_data

# 从模型中自动解析障碍物参数
def parse_obstacles_from_model():
    pillars_xy = []
    pillar_heights = []
    pillar_radius = None
    wall_y_max = None
    wall_x_lim = None

    # 遍历所有几何体
    for i in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if geom_name is None:
            continue

        geom_type = model.geom_type[i]
        geom_size = model.geom_size[i]
        body_id = model.geom_bodyid[i]
        body_pos = model.body_pos[body_id]


        if geom_name.startswith('pillar') and geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            pillars_xy.append([body_pos[0], body_pos[1]])
            pillar_heights.append(body_pos[2] * 2)  
            if pillar_radius is None:
                pillar_radius = geom_size[0]  

        elif geom_name.startswith('wall') and geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            if 'wall_y' in geom_name:
                wall_y_max = body_pos[1] - geom_size[1]  
            elif 'wall_x' in geom_name:
                wall_x_lim = abs(body_pos[0]) - geom_size[0]  

    return (np.array(pillars_xy) if pillars_xy else np.array([[0.4, 0.4], [-0.4, 0.5], [0.2, 0.6]]),
            np.array(pillar_heights) if pillar_heights else np.array([0.8, 0.7, 1.0]),
            pillar_radius if pillar_radius else 0.06,
            wall_y_max if wall_y_max else 1.1,
            wall_x_lim if wall_x_lim else 1.1)


PILLARS_XY, PILLAR_HEIGHTS, _pillar_r, WALL_Y_MAX, WALL_X_LIM = parse_obstacles_from_model()
SAFETY_MARGIN = 0.05
PILLAR_R = _pillar_r + 0.04 + SAFETY_MARGIN  

print(f"Parsed obstacles from XML:")
print(f"  Pillars XY: {PILLARS_XY.tolist()}")
print(f"  Pillar heights: {PILLAR_HEIGHTS.tolist()}")
print(f"  Pillar radius: {_pillar_r:.3f}")
print(f"  Wall Y max: {WALL_Y_MAX:.3f}")
print(f"  Wall X limit: {WALL_X_LIM:.3f}")

LINK_NAMES = ["base", "link1", "link2", "link3", "link4", "link5", "link6"]
LINK_RADII  = [0.08,   0.04,   0.035,  0.03,   0.025,  0.02,   0.03]
ARM_BODY_IDS = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in LINK_NAMES]
EE_SITE_ID   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

JOINT_LIMITS = np.array([[-2.618,2.618],[-2.094,2.094],[-2.967,2.967],
                          [-1.832,1.832],[-1.22,1.22],[-3.14,3.14]])

def _seg_point_dist_xy(p, q, c):
    pq = q - p
    t = np.dot(c - p, pq) / (np.dot(pq, pq) + 1e-12)
    t = np.clip(t, 0, 1)
    closest = p + t * pq
    return np.linalg.norm(closest - c)

def is_config_collision_free(qpos):
    d = get_ik_data()
    d.qpos[:] = qpos
    mujoco.mj_forward(model, d)
    pts = d.xpos[ARM_BODY_IDS]  

    if np.any(pts[:, 1] > WALL_Y_MAX) or np.any(np.abs(pts[:, 0]) > WALL_X_LIM):
        return False


    for i in range(len(pts) - 1):
        p, q = pts[i, :2], pts[i+1, :2]  
        r = LINK_RADII[i] + SAFETY_MARGIN
        z_lo = min(pts[i, 2], pts[i+1, 2])
        z_hi = max(pts[i, 2], pts[i+1, 2])
        for j, (cx, cy) in enumerate(PILLARS_XY):
            if z_lo > PILLAR_HEIGHTS[j] or z_hi < 0:
                continue
            dist_xy = _seg_point_dist_xy(p, q, np.array([cx, cy]))
            if dist_xy < PILLAR_R + r - SAFETY_MARGIN:
                return False
    return True

def is_edge_collision_free(q1, q2):
    dist = np.linalg.norm(q2 - q1)
    steps = max(5, int(dist / 0.05))
    for i in range(1, steps + 1):
        if not is_config_collision_free(q1 + (q2 - q1) * i / steps):
            return False
    return True

def ik_dls(target_pos, qpos_init, tol=1e-3, max_iter=300, damping=5e-2, step=0.3):
    jacp = np.zeros((3, model.nv))
    d = get_ik_data()
    d.qpos[:] = qpos_init
    for _ in range(max_iter):
        mujoco.mj_forward(model, d)
        err = target_pos - d.site_xpos[EE_SITE_ID]
        if np.linalg.norm(err) < tol:
            return d.qpos.copy(), True
        mujoco.mj_jacSite(model, d, jacp, None, EE_SITE_ID)
        dq = jacp.T @ np.linalg.solve(jacp @ jacp.T + damping**2 * np.eye(3), err)
        mujoco.mj_integratePos(model, d.qpos, dq, step)
    return d.qpos.copy(), False

def ik_dls_multi(target_pos, n_seeds=40, extra_hints=None):
    hints = list(extra_hints) if extra_hints else []
    hints.append(np.zeros(model.nq))
    hints += [np.array([np.random.uniform(lo, hi) for lo, hi in JOINT_LIMITS])
              for _ in range(n_seeds)]
    for q0 in hints:
        q, ok = ik_dls(target_pos, q0)
        if ok and is_config_collision_free(q):
            return q, True

    best_q, best_err = None, np.inf
    for q0 in hints[:10]:
        q, _ = ik_dls(target_pos, q0)
        if not is_config_collision_free(q):
            continue
        d = get_ik_data()
        d.qpos[:] = q
        mujoco.mj_forward(model, d)
        err = np.linalg.norm(target_pos - d.site_xpos[EE_SITE_ID])
        if err < best_err:
            best_err, best_q = err, q.copy()
    if best_q is not None:
        return best_q, best_err < 0.05
    return hints[0], False

def _rrtc_extend(nodes, nodes_arr, parents, q_target, step_size):
    dists = np.linalg.norm(nodes_arr[:len(nodes)] - q_target, axis=1)
    near_idx = int(np.argmin(dists))
    q_near = nodes[near_idx]
    diff = q_target - q_near
    dist = np.linalg.norm(diff)
    q_new = q_near + diff / (dist + 1e-9) * min(step_size, dist)
    if not is_config_collision_free(q_new) or not is_edge_collision_free(q_near, q_new):
        return -1, False
    idx = len(nodes)
    nodes.append(q_new)
    nodes_arr[idx] = q_new
    parents.append(near_idx)
    return idx, np.linalg.norm(q_new - q_target) < step_size

def rrt_plan(goal_pos, start_qpos, max_iter=6000, step_size=0.2):
    goal_q, ok = ik_dls_multi(goal_pos, n_seeds=40, extra_hints=[start_qpos])
    if not ok:
        print("IK failed for goal.")
        return None
    if not is_config_collision_free(goal_q):
        print("Goal config in collision.")
        return None

    cap = max_iter + 2
    ta_nodes, ta_parents = [start_qpos.copy()], [-1]
    tb_nodes, tb_parents = [goal_q.copy()], [-1]
    ta_arr = np.empty((cap, model.nq)); ta_arr[0] = ta_nodes[0]
    tb_arr = np.empty((cap, model.nq)); tb_arr[0] = tb_nodes[0]
    ta_conn = tb_conn = -1

    for _ in range(max_iter):
        q_rand = (tb_nodes[0] if np.random.random() < 0.1
                  else np.array([np.random.uniform(lo, hi) for lo, hi in JOINT_LIMITS]))
        idx_a, _ = _rrtc_extend(ta_nodes, ta_arr, ta_parents, q_rand, step_size)
        if idx_a == -1:
            continue
        idx_b, reached = _rrtc_extend(tb_nodes, tb_arr, tb_parents, ta_nodes[idx_a], step_size)
        if idx_b == -1:
            continue
        if reached:
            ta_conn, tb_conn = idx_a, idx_b
            break
        ta_nodes, tb_nodes = tb_nodes, ta_nodes
        ta_parents, tb_parents = tb_parents, ta_parents
        ta_arr, tb_arr = tb_arr, ta_arr

    if ta_conn == -1:
        print("RRTConnect failed.")
        return None

    path_a, idx = [], ta_conn
    while idx != -1:
        path_a.append(ta_nodes[idx]); idx = ta_parents[idx]
    path_a.reverse()
    path_b, idx = [], tb_conn
    while idx != -1:
        path_b.append(tb_nodes[idx]); idx = tb_parents[idx]

    path = path_a + path_b
    print(f"Path found: {len(path)} waypoints.")
    return path

def smooth_path(path):
    for _ in range(300):
        if len(path) <= 2:
            break
        i = np.random.randint(0, len(path) - 2)
        j = np.random.randint(i + 2, min(i + 10, len(path)))
        if is_edge_collision_free(path[i], path[j]):
            path = path[:i+1] + path[j:]
    return path

def interpolate_path(waypoints, max_joint_vel=0.04):
    dt = model.opt.timestep
    segs = []
    for i in range(len(waypoints) - 1):
        q1, q2 = np.array(waypoints[i]), np.array(waypoints[i+1])
        max_diff = np.max(np.abs(q2 - q1))
        n = max(2, int(max_diff / (max_joint_vel * dt)))
        t = np.linspace(0, 1, n, endpoint=False)
        s = t * t * (3 - 2 * t)  
        segs.append(q1 + np.outer(s, q2 - q1))
    segs.append(waypoints[-1:])
    return list(np.concatenate(segs))

cart_targets = np.array([
    [ 0.5,  0.1, 1.0],   
    [ 0.7,  0.3, 0.6],   
    [-0.6,  0.2, 0.9],   
    [ 0.0,  0.3, 0.5],   
])

data.qpos[:] = 0
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
