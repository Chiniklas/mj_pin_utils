from typing import List
import os
import numpy as np
import argparse
import mujoco

from mj_pin.abstract import DataRecorder, VisualCallback
from mj_pin.simulator import Simulator
from mj_pin.utils import PinController, get_robot_description, mj_frame_pos

class PDController(PinController):
    def __init__(self, urdf_path, Kp = 44., Kd = 3.):
        super().__init__(urdf_path, floating_base_quat=True)
        self.Kp, self.Kd = Kp, Kd
        self.q0 = None
    
    def get_torques(self, sim_step, mj_data):
        # Get state in pinocchio format
        q, v = self.get_state(mj_data)
        qj, vj = q[-self.nu:], v[-self.nu:]

        if self.q0 is None: self.q0 = qj.copy()

        torques_array = self.Kp * (self.q0 - qj) - self.Kd * vj
        # {joint name : torque}
        torque_map = self.create_torque_map(torques_array)

        return torque_map
    
class FeetVisualCallback(VisualCallback):
    def __init__(self, xml_path, feet_names : List[str], update_step = 1):
        super().__init__(update_step)
        self.xml_path = xml_path
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.feet_names = feet_names

    def add_visuals(self, mj_data):
        radius = 0.03
        for i, f_name in enumerate(self.feet_names):
            pos = mj_frame_pos(self.mj_model, mj_data, f_name)
            self.add_sphere(pos, radius, self.colors.id(i))

class StateDataRecorder(DataRecorder):
    def __init__(
        self,
        record_dir: str = "",
        record_step: int = 1,
    ) -> None:
        """
        A simple data recorder that saves simulation data to a .npz file.
        """
        super().__init__(record_dir, record_step)
        self.data = {}
        self.reset()

    def reset(self) -> None:
        """
        Reset the recorder data.
        """
        self.data = {"time": [], "q": [], "v": [], "ctrl": [],}

    def save(self) -> None:
        """
        Save the recorded data to a file in the specified directory.
        """
        if not self.record_dir:
            self.record_dir = os.getcwd()
        # Uncomment to save data
        # os.makedirs(self.record_dir, exist_ok=True)

        timestamp = self.get_date_time_str()
        file_path = os.path.join(self.record_dir, f"simulation_data_{timestamp}.npz")

        try:
            print(self.data["time"][:10])
            # Uncomment to save data
            # np.savez(file_path, **self.data)
            print(f"Data successfully saved to {file_path}")
        except Exception as e:
            print(f"Error saving data: {e}")

    def _record(self, mj_data) -> None:
        """
        Record simulation data at the current simulation step.

        Args:
            sim_step (int): Current simulation step.
            mj_data (Any): MuJoCo data object.
            **kwargs: Additional data to record.
        """
        # Record time and state
        self.data["time"].append(round(mj_data.time, 4))
        self.data["q"].append(mj_data.qpos.copy())
        self.data["v"].append(mj_data.qvel.copy())
        self.data["ctrl"].append(mj_data.ctrl.copy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate a robot with optional recording and visualization.")
    parser.add_argument("--record_video", action="store_true", help="Record a video of the simulation.")
    parser.add_argument("--record_data", action="store_true", help="Record simulation data.")
    parser.add_argument("--robot_name", type=str, default="go2", help="Name of the robot to simulate.")
    args = parser.parse_args()

    # Load robot information and paths
    robot_description = get_robot_description(args.robot_name)
    robot_description.eeff_frame_name = ["FL", "FR", "RL", "RR"]

    # Load the simulator
    # Will start two threads: one for the viewer and one for the physics simulation.
    # Viewer and physics are updated at different rate.
    sim = Simulator(robot_description.xml_scene_path, sim_dt=1e-3, viewer_dt=1/40)

    # PD Controller, called every simulation step
    pd_controller = PDController(robot_description.urdf_path)

    # Visual callback on the viewer, called every viewer step
    vis_feet_pos = FeetVisualCallback(robot_description.xml_path, robot_description.eeff_frame_name)
    
    # Data recorder, called every 10 simulation step
    record_state_data = None
    if args.record_data:
        record_state_data = StateDataRecorder("./data", record_step=10)

    # Set video setting to high quality
    if args.record_video:
        sim.vs.set_high_quality()

    # Add a box in the scene
    sim.edit.add_box(
        pos=[0., 0., 0.1],
        size=[0.1, 0.1, 0.1],
        euler=[0., 0., 0.3],
        color="black",
        name="box_under"
    )
    
    # Run the simulation with the provided controller etc.
    sim.run(controller=pd_controller,
            data_recorder=record_state_data,
            visual_callback=vis_feet_pos,
            record_video=args.record_video)