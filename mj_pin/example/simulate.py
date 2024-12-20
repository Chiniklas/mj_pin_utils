from typing import List
import os
import numpy as np

from mj_pin.abstract import DataRecorder, VisualCallback
from mj_pin.simulator import Simulator
from mj_pin.utils import PinController, load_mj_pin, mj_frame_pos

class PDController(PinController):
    def __init__(self, pin_model, Kp = 44., Kd = 3.):
        super().__init__(pin_model)
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
    def __init__(self, mj_model, feet_names : List[str], update_step = 1):
        super().__init__(update_step)
        self.mj_model = mj_model
        self.feet_names = feet_names

    def _add_visuals(self, mj_data):
        radius = 0.03
        for i, f_name in enumerate(self.feet_names):
            pos = mj_frame_pos(self.mj_model, mj_data, f_name)
            self.add_sphere(pos, radius, self.colors_id[i])

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
        # os.makedirs(self.record_dir, exist_ok=True)

        timestamp = self.get_date_time_str()
        file_path = os.path.join(self.record_dir, f"simulation_data_{timestamp}.npz")

        try:
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
        self.data["time"].append(mj_data.time)
        self.data["q"].append(mj_data.qpos)
        self.data["v"].append(mj_data.qvel)
        self.data["ctrl"].append(mj_data.ctrl)


if __name__ == "__main__":
    mj_model, pin_model, robot_description = load_mj_pin("go2")
    pd_controller = PDController(pin_model)
    vis_feet_pos = FeetVisualCallback(mj_model, robot_description.eeff_frame_name)
    record_state_data = StateDataRecorder("./data", record_step=10)

    sim = Simulator(mj_model)
    sim.run(controller=pd_controller,
            data_recorder=record_state_data,
            visual_callback=vis_feet_pos)