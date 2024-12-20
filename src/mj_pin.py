from typing import Dict, List, Tuple
import mujoco.specs_test
import numpy as np
import mujoco
import pinocchio as pin
import os

from mj_pin.mj_utils import load_mj
from mj_pin.abstract import Controller
from mj_pin.utils.transform import get_skew_sim_mat

MJ2PIN_QUAT = [1,2,3,0]
PIN2MJ_QUAT = [3,0,1,2]

def load_mj_pin(
        robot_name : str,
        with_floor : bool = True,
        copy_motor_param : bool = True
        ):
    # Get robot model file path
    mj_model, desc = load_mj(robot_name, with_floor)
    # Path without scene
    path_mjcf = desc.model_path

    # Load model (without scene)
    pin_model = pin.buildModelFromMJCF(path_mjcf)

    # Update pinocchio model
    if desc:
        desc.model_path = path_mjcf
        # Add end effecor frames from description to pinocchio model
        add_frames_from_mujoco(pin_model, mj_model, desc.eeff_frame_name)

    # Update motor paramters (damping, friction, armature)
    if copy_motor_param:
        copy_motor_parameters(pin_model, mj_model)

    return mj_model, pin_model, desc


def pin_joint_name2id(pin_model) -> Dict[str, int]:
    """
    Init joint name to id map.
    """
    pin_n_joints = len(pin_model.joints)
    pin_joint_name2id = {
        pin_model.names[i] : i
        for i in range(pin_n_joints)
        if (# Only 1 DoF joints (no root joint)
            pin_model.joints[i].nq == 1 and 
                # No universe joint or ill-defined joints
            pin_model.joints[i].id <= pin_n_joints)
    }
    return pin_joint_name2id

def pin_joint_name2act_id(pin_model) -> Dict[str, int]:
    # Get joint id to name map sorted by index
    joint_name2id = pin_joint_name2id(pin_model)
    pin_joint_id2name = {
        i : name
        for name, i
        in joint_name2id.items() 
    }

    # Map to joints
    pin_joint_name2act_id = {
        name : i 
        for i, name in enumerate(pin_joint_id2name.values())
    }
    return pin_joint_name2act_id

def pin_frame_pos(pin_model, pin_data, frame_name: str) -> np.ndarray:
    """
    Get the frame position in base frame for a given frame name in Pinocchio.

    Args:
        frame_name (str): Name of the frame.

    Returns:
        np.ndarray: Position of the frame in the world frame.
    """
    frame_id = pin_model.getFrameId(frame_name)
    if frame_id >= len(pin_model.frames):
        raise ValueError(f"Frame '{frame_name}' not found in the Pinocchio model.")
    
    # Get frame position in the world frame
    frame_pos = pin_data.oMf[frame_id].translation
    return frame_pos

def mj_2_pin_state(q_wxyz: np.ndarray) -> np.ndarray:
    """
    Convert MuJoCo to Pinocchio state format.
    Convert quaternion format:
    qw, qx, qy, qz -> qx, qy, qz, qw
    """
    q_xyzw = q_wxyz
    q_xyzw[3:7] = np.take(
        q_wxyz[3:7],
        MJ2PIN_QUAT,
        mode="clip",
        )
    return q_xyzw

def pin_2_mj_state(q_xyzw: np.ndarray) -> np.ndarray:
    """
    Convert MuJoCo to Pinocchio state format.
    Convert quaternion format:
    qx, qy, qz, qw -> qw, qx, qy, qz
    """
    q_wxyz = q_xyzw
    q_wxyz[3:7] = np.take(
        q_xyzw[3:7],
        PIN2MJ_QUAT,
        mode="clip",
        )
    return q_wxyz

def mj_2_pin_qv(q_mj : np.ndarray, v_mj : np.ndarray) -> np.ndarray:
    """
    Convert MuJoCo to Pinocchio state and velocities format.
    qw, qx, qy, qz -> qx, qy, qz, qw
    lin vel : global -> local
    angular vel : local -> local (no change)
    """
    q_pin = mj_2_pin_state(q_mj)
    # Transform from world to base
    b_T_W = pin.XYZQUATToSE3(q_pin[:7]).inverse()
    R = b_T_W.rotation
    p = b_T_W.translation
    p_skew = get_skew_sim_mat(p)

    # v_b = [p] @ R @ w_W + R @ v_W
    #     = [p] @ R @ R.T @ w_B + R @ v_W
    #     = [p] @ w_B + R @ v_W
    v_mj[:3] = p_skew @ v_mj[3:6] + R @ v_mj[:3]
    return q_pin, v_mj

def pin_2_mj_qv(q_pin : np.ndarray, v_pin : np.ndarray) -> np.ndarray:
    """
    Convert Pinocchio to MuJoCo state and velocities format.
    qx, qy, qz, qw -> qw, qx, qy, qz
    lin vel : local -> global
    angular vel : local -> local (no change)
    """
    q_mj = pin_2_mj_state(q_mj)
    # Transform from world to base
    W_T_b = pin.XYZQUATToSE3(q_pin[:7])
    R = W_T_b.rotation
    p = W_T_b.translation
    p_skew = get_skew_sim_mat(p)

    # v_W = [p] @ R @ w_b + R @ v_b
    v_pin[:3] = p_skew @ R @ v_pin[3:6] + R @ v_pin[:3]
    return q_mj, v_pin


class PinController(Controller):

    def __init__(self, pin_model):
        super().__init__()
        self.pin_model = pin_model
        self.pin_data = pin.Data(pin_model)

        self.nq = pin_model.nq
        self.nv = pin_model.nv
        self.nu = len([j for j in pin_model.joints 
                       if j.id < self.nv and j.nv == 1])

        self.joint_name2act_id = pin_joint_name2act_id(self.pin_model)

    def get_state(self, mj_data) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get state in pinocchio format from mujoco data.

        q : [
             x, y, z,
             qx, qy, qz, qw
             j1, ..., je,
            ]
        
        v : [
             vx, vy, vz, (local frame)
             wx, wy, wz, (local frame)
             vj1, ..., vje,
            ]

        Returns:
            Tuple[np.ndarray, np.ndarray]: q [nq], v [nv]
        """
        q_pin, v_pin = mj_2_pin_qv(mj_data.qpos.copy(), mj_data.qvel.copy())

        return q_pin, v_pin
        
    def create_torque_map(self, torques : np.ndarray) -> Dict[str, float]:
        torque_map = {
            j_name : torques[joint_id]
            for j_name, joint_id
            in self.joint_name2act_id.items()
        }
        return torque_map

def add_frames_from_mujoco(
    pin_model, 
    mj_model, 
    frame_geometries: List[str]
) -> None:
    """
    Add frames to a Pinocchio model based on geometries defined in a MuJoCo model,
    using the parent joint names.

    Args:
        pin_model (pin.Model): The Pinocchio model to which frames will be added.
        mj_model (mujoco.MjModel): The MuJoCo model from which geometries will be retrieved.
        frame_geometries (List): List of geometry names in MuJoCo to be added as frames in Pinocchio.
    """
    for geom_name in frame_geometries:
        # Get geometry ID in MuJoCo
        geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id == -1:
            print(f"Geometry '{geom_name}' not found in MuJoCo model. Skipping.")
            continue

        # Get parent joint of the geometry
        parent_joint_id = mj_model.body_jntadr[mj_model.geom_bodyid[geom_id]]
        parent_joint_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, parent_joint_id)

        # Find corresponding Pinocchio joint ID
        if parent_joint_name not in pin_model.names:
            print(f"Joint '{parent_joint_name}' not found in Pinocchio model. Skipping.")
            continue
        pin_joint_id = pin_model.getJointId(parent_joint_name)

        # Get the position and orientation of the geometry in the body frame
        geom_pos = mj_model.geom_pos[geom_id]
        geom_quat = mj_model.geom_quat[geom_id]  # Quaternion in (w, x, y, z) format
        geom_rotation = pin.Quaternion(geom_quat[0], *geom_quat[1:]).toRotationMatrix()

        # Create SE3 transformation
        geom_to_joint = pin.SE3(geom_rotation, geom_pos)

        # Add the frame to the Pinocchio model
        frame_name = f"{geom_name}"
        new_frame = pin.Frame(
            frame_name,
            pin_joint_id,
            pin_joint_id,
            geom_to_joint,
            pin.FrameType.OP_FRAME,
        )
        pin_model.addFrame(new_frame)

        print(f"Added frame '{frame_name}' to Pinocchio model.")

def copy_motor_parameters(pin_model, mj_model: mujoco.MjModel) -> None:
    """
    Update motor parameters (friction, damping, and rotor inertia) in a Pinocchio model
    from a MuJoCo model.

    Args:
        pin_model (pin.Model): The Pinocchio model to update.
        mj_model (mujoco.MjModel): The MuJoCo model to use as a reference.
    """
    # Update friction
    pin_model.friction = mj_model.dof_frictionloss.copy()
    # Update damping
    pin_model.damping = mj_model.dof_damping.copy()
    # Update armature
    pin_model.rotorInertia = mj_model.dof_armature.copy()

    # Update kinematic limits
    nu = mj_model.nu

    # Update position limits
    pin_model.upperPositionLimit = mj_model.jnt_range[:, 1].copy()
    pin_model.lowerPositionLimit = mj_model.jnt_range[:, 0].copy()

    # Update effort limits
    pin_model.effortLimit[-nu:] = np.abs(np.max(np.hstack((
        mj_model.actuator_ctrlrange[:, 0],
        mj_model.actuator_ctrlrange[:, 1],
    )), axis=0))
