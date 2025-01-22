import time
from typing import Optional
import mujoco.memory_leak_test
import numpy as np
import mujoco
import mujoco.viewer
from threading import Thread
import threading
from functools import wraps
from dataclasses import dataclass
import cv2
import os
from datetime import datetime

from mj_pin.abstract import Controller, DataRecorder, VisualCallback
from mj_pin.model_editor import ModelEditor
from mj_pin.utils import pin_2_mj_qv, mj_joint_name2act_id

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t_start = time.time()
        try:
            result = func(*args, **kwargs)
        finally:
            elapsed_time = time.time() - t_start
        return elapsed_time if result is None else (elapsed_time, result)
    return wrapper

@dataclass
class VideoSettings:
    """
    Video recording settings for the simulator.
    """
    video_dir: str = "./camera_recording/"
    width: int = 640
    height: int = 480
    fps: int = 24
    playback_speed: float = 1.0
    track_obj : str = ""
    distance : float = 2.0
    azimuth : float = 140.
    elevation : float = -30.0

    def set_top_view(self):
        self.azimuth = 0
        self.elevation = -90

    def set_front_view(self):
        self.azimuth = 0
        self.elevation = 0

    def set_bottom_view(self):
        self.azimuth = 0
        self.elevation = 90

    def set_side_view(self):
        self.azimuth = 90
        self.elevation = 0
    
    def set_high_quality(self):
        self.width = 1440
        self.height = 1024

    def set_low_quality(self):
        self.width = 640
        self.height = 480

class Simulator:
    def __init__(
        self,
        xml_path : str,
        sim_dt : float = 1.0e-3,
        viewer_dt : float = 1/40.,
        ):
        self.sim_dt = sim_dt
        self.viewer_dt = viewer_dt
        self.vs = VideoSettings()

        # Model editor
        self.edit = ModelEditor(xml_path)

        # Threading for physics and viewer
        self.__locker = threading.RLock()

    def _init_model_data(self) -> None:
        self.mj_model = self.edit.get_model()
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_dt
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)
        self.joint_name2act_id = mj_joint_name2act_id(self.mj_model)
    
    @staticmethod
    def get_date_time_str() -> str:
        now = datetime.now()
        date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
        return date_time

    def reset(
        self,
        q : np.ndarray = None,
        v : np.ndarray = None,
        pin_format : bool = False) -> None:
        
        # Reset to keyframe
        if q is None and v is None:
            mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)

        v = np.zeros(self.mj_model.nv) if v is None else v

        if pin_format: 
            q, v = pin_2_mj_qv(q, v)
        self._set_state(q, v)

    def set_video_settings(self, video_settings : VideoSettings) -> None:
        self.vs = video_settings
        self.mj_model.visual.offwidth = self.vs.width
        self.mj_model.visual.offheight = self.vs.height

    def _set_state(self, q_mj : np.ndarray, v_mj : np.ndarray):
        self.mj_data.qpos = q_mj.copy()
        self.mj_data.qvel = v_mj.copy()
        self.mj_data.ctrl = np.zeros(self.mj_model.nu)
        mujoco.mj_forward(self.mj_model, self.mj_data)       

    def setup(self) -> None:
        # Variables
        self.sim_step : int = 0
        self.viewer_step : int = 0
        self.time : float = 0.
        self.stop_sim = False
        self.use_viewer = False
        self.controller = None
        self.data_recorder = None
        self.visual_callback = None
        # Record video
        self.rendering_cam = None
        self.renderer = None
        self.frames = []
        # Init model and data
        self._init_model_data()

    def setup_camera_recording(self) -> None:
        if self.mj_model.vis.global_.offwidth < self.vs.width:
            self.mj_model.vis.global_.offwidth = self.vs.width

        if self.mj_model.vis.global_.offheight < self.vs.height:
            self.mj_model.vis.global_.offheight = self.vs.height

        renderer = mujoco.Renderer(self.mj_model, self.vs.height, self.vs.width)
        self.rendering_cam = mujoco.MjvCamera()
        if self.vs is None: self.vs = VideoSettings()

        return renderer 

    def _update_camera_position(self, viewer) -> None:

        # Take the position from viewer if available
        if self.use_viewer:
            if viewer:
                self.rendering_cam.distance = viewer.cam.distance
                self.rendering_cam.azimuth = viewer.cam.azimuth
                self.rendering_cam.elevation = viewer.cam.elevation
                self.rendering_cam.lookat = viewer.cam.lookat
        else:
            # Update camera position
            self.rendering_cam.distance, self.rendering_cam.azimuth, self.rendering_cam.elevation =\
                (self.vs.distance,
                 self.vs.azimuth,
                 self.vs.elevation
                 )
            
        # Track object
        if self.vs.track_obj:
            obj_pose = np.zeros(3)
            # Geom
            obj_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, self.vs.track_obj)
            if obj_id > 0:
                obj_pose = self.mj_data.geom_xpos[obj_id].copy()
            # Body
            else:
                obj_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.vs.track_obj)
                if obj_id > 0:
                    obj_pose = self.mj_data.xpos[obj_id].copy()

            self.rendering_cam.lookat = obj_pose

    def _control_step(self) -> None:
        # joint name : torque value
        torque_map = self.controller.get_torques(self.sim_step, self.mj_data)

        torque_ctrl = np.zeros(self.mj_model.nu)
        for joint_name, torque_value in torque_map.items():
            torque_ctrl[
                self.joint_name2act_id[joint_name]
                ] = torque_value

        self.mj_data.ctrl = torque_ctrl

    def _record_data_step(self) -> None:
        self.data_recorder.record(self.sim_step, self.mj_data)

    @timing_decorator
    def _physics_step(self) -> None:
        # Compute state, vel
        mujoco.mj_step1(self.mj_model, self.mj_data)

        # Compute torques and set torques
        if self.controller is not None:
            self._control_step()

        # TODO: Apply external forces

        # Apply control
        mujoco.mj_step2(self.mj_model, self.mj_data)
        
        self.sim_step += 1
        self.time += self.sim_dt

    def _run_physics(self):
        if self.record_video:
            with self.__locker:
                renderer = self.setup_camera_recording()

        while not(self.stop_sim):
            self._stop_sim()
            self.__locker.acquire()
            physics_time = self._physics_step()

            # Record data
            if self.data_recorder is not None:
                self._record_data_step()

            # Record video
            if self.record_video:
                self._record_frame(renderer, None)

            self.__locker.release()

            if self.use_viewer:
                sleep_time = self.sim_dt - physics_time
                if sleep_time > 0.:
                    time.sleep(sleep_time)

        mujoco.mj_forward(self.mj_model, self.mj_data)
        if self.record_video:
            renderer.close()

    @timing_decorator
    def _viewer_step(self, viewer) -> float:
        viewer.sync()
        self.viewer_step += 1

    def _record_frame(self, renderer, viewer) -> float:
        if len(self.frames) < self.mj_data.time * self.vs.fps * self.vs.playback_speed:
            self._update_camera_position(viewer)
            renderer.update_scene(self.mj_data, self.rendering_cam)
            pixels = renderer.render()
            self.frames.append(pixels)

    def _run_viewer(self):
        with self.__locker:
            viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data, show_left_ui=False, show_right_ui=False)
        
        while viewer.is_running():
            if self.stop_sim:
                break

            with self.__locker:
                render_time = self._viewer_step(viewer)
            # Update visual
            if self.visual_callback:
                self._update_visual(viewer)
            # Update camera position
            if self.record_video:
                self._update_camera_position(viewer)

            sleep_time = self.viewer_dt - render_time
            if sleep_time > 0.:
                time.sleep(sleep_time)

        self.stop_sim = True
        viewer.close()

    def _stop_sim(self) -> None:
        if self.sim_time > 0 and self.sim_step * self.sim_dt >= self.sim_time:
            self.stop_sim = True
        
        if self.controller and self.controller.diverged:
            self.stop_sim = True

    def _start_viewer(self,):
        viewer_thread = Thread(target=self._run_viewer)
        viewer_thread.start()

        return viewer_thread
    
    def _update_visual(self, viewer):
        self.visual_callback.render(
            self.sim_step,
            viewer,
            self.mj_data
            )

    def save_video(self, save_path: str) -> None:        
        if not self.frames:
            print("No frames recorded.")
            return
        
        # Check save path
        dirname, filename = os.path.split(save_path)
        if not filename:
            filename = "video_" + self.get_date_time_str() + ".mp4"
        if not dirname:
            dirname = "./"
        os.makedirs(dirname, exist_ok=True)
        save_path = os.path.join(dirname, filename)

        # Check extension
        _, ext = os.path.splitext(save_path)
        VALID_EXT = ".mp4"
        if ext != VALID_EXT:
            save_path += VALID_EXT

        # Create video from frames
        height, width, _ = self.frames[0].shape
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), self.vs.fps, (width, height))
        for frame in self.frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()

        print(f"Video saved to {save_path}")

    def run(self,
            sim_time : float = 0.,
            use_viewer : bool = True, 
            controller : Controller = None,
            data_recorder : DataRecorder = None,
            visual_callback : VisualCallback = None,
            record_video : bool = False,
            ):
        # Init simulator
        self.setup()
        self.sim_time = sim_time
        self.use_viewer = use_viewer
        self.record_video = record_video
        self.controller = controller
        self.data_recorder = data_recorder
        self.visual_callback = visual_callback
        if self.data_recorder: data_recorder.reset()

        # Start viewer thread          
        viewer_thread = None
        if use_viewer:
            viewer_thread = Thread(target=self._run_viewer)
            viewer_thread.start()

        # Start physics thread
        physics_thread = Thread(target=self._run_physics)
        physics_thread.start()

        try:
            # Wait for threads to complete
            physics_thread.join()
            if viewer_thread: viewer_thread.join()

        except KeyboardInterrupt:
            print("\nSimulation interrupted.")
            self.stop_sim = True

            if physics_thread.is_alive():
                physics_thread.join()
            # Ensure threads are stopped
            if viewer_thread and viewer_thread.is_alive():
                viewer_thread.join()

        print("Simulation stopped.")

        if self.data_recorder:
            self.data_recorder.save()

        if self.record_video:
            self.save_video(self.vs.video_dir)
        
    def visualize_trajectory(self,
        joint_traj: np.ndarray,
        time_traj: Optional[np.ndarray] = None,
        record_video : bool = False,
        ) -> None:
        """
        Visualize a joint trajectory using the MuJoCo viewer.

        Args:
            joint_traj (np.ndarray): The joint trajectory to visualize. Shape: (N, nq).
            time_traj (np.ndarray): The corresponding time trajectory. Shape: (N,).
            record_video (bool): Record video of the trajectory
        """
        self.setup()

        N = len(joint_traj)
        if time_traj is None:
            time_traj = np.linspace(0, N * self.sim_dt, N)

        assert joint_traj.shape[0] == len(time_traj), \
            "The number of trajectory points must match the time trajectory length."
        assert joint_traj.shape[1] == self.mj_model.nq, \
            f"The trajectory dimension must match the model's nq ({self.mj_model.nq})."

        dt_traj = np.diff(time_traj, append=0.)

        if record_video:
            renderer = self.setup_camera_recording()

        print(f"Visualizing trajectory...")
        self.use_viewer = True
        try:
            with mujoco.viewer.launch_passive(self.mj_model, self.mj_data, show_left_ui=False, show_right_ui=False) as viewer:
                while viewer.is_running():
                    self.mj_data.time = 0.
                    self.viewer_step = 0
                    viewer_time = 0.
                    # Iterate over state trajectory
                    for qpos, dt in zip(joint_traj, dt_traj):
                        if not viewer.is_running():
                            break

                        # Set the state
                        self.mj_data.qpos[:] = qpos
                        mujoco.mj_forward(self.mj_model, self.mj_data)

                        # Record video
                        if record_video:
                            self._update_camera_position(viewer)
                            self._record_frame(renderer, viewer)

                        # Update the viewer
                        render_time = 0.
                        if viewer_time <= self.mj_data.time:
                            render_time = self._viewer_step(viewer)

                        # Update time
                        self.mj_data.time += dt
                        viewer_time = self.viewer_step * self.viewer_dt

                        # Sleep to match the time trajectory
                        sleep_time = max(dt - render_time, 0.)
                        time.sleep(sleep_time)

                    # Sleep one second before starting again
                    if viewer.is_running():
                        time.sleep(1)

        except KeyboardInterrupt:
            print("\nTrajectory visualization interrupted.")

        finally:
            if record_video:
                renderer.close()
                self.save_video(self.vs.video_dir)
            print("Trajectory visualization complete.")


if __name__ == "__main__":
    from mj_pin.utils import load_mj

    _, robot_description = load_mj("go2")
    sim = Simulator(robot_description.scene_path)
    sim.vs.track_obj = "base"
    sim.vs.set_side_view()
    sim.run(use_viewer=False, record_video=True)