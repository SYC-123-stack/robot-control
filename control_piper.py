import mujoco
import mujoco.viewer
import numpy as np

model_path = "./scene.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

home_pose = np.array([0, 1.57, -1.3485, 0, 0, 0, 0, 0])
wave_pose1 = np.array([0.5, 1.57, -1.3485, 0, 0.5, 0, 0, 0])
wave_pose2 = np.array([-0.5, 1.57, -1.3485, 0, -0.5, 0, 0, 0])
gripper_open = np.array([0, 1.57, -1.3485, 0, 0, 0, 0, 0])
gripper_close = np.array([0, 1.57, -1.3485, 0, 0, 0, 0.03, -0.03])

data.qpos[:] = home_pose
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    phase_duration = 2000

    while viewer.is_running():
        current_step_in_phase = step % phase_duration

        if step < phase_duration:
            target_pose = home_pose
        elif step < 2 * phase_duration:
            target_pose = wave_pose1
        elif step < 3 * phase_duration:
            target_pose = wave_pose2
        elif step < 4 * phase_duration:
            target_pose = home_pose
        elif step < 5 * phase_duration:
            target_pose = gripper_open
        elif step < 6 * phase_duration:
            target_pose = gripper_close
        else:
            step = 0
            continue

        data.ctrl[:] = target_pose[:7]
        mujoco.mj_step(model, data)
        viewer.sync()
        step += 1

