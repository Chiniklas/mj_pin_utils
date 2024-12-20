import time
import numpy as np
import mujoco
import mujoco.viewer
from threading import Thread
import threading
from functools import wraps

from mj_pin.abstract import Controller, DataRecorder, VisualCallback
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

class Simulator:
    def __init__(
        self,
        mj_model,
        sim_dt : float = 1.0e-3,
        viewer_dt : float = 1/50.,
        ):
        self.sim_dt = sim_dt
        self.viewer_dt = viewer_dt

        self.mj_model = mj_model
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)

        self.joint_name2act_id = mj_joint_name2act_id(self.mj_model)

        # Threading for physics and viewer
        self.locker = threading.Lock()

    def reset(
        self,
        q : np.ndarray = None,
        v : np.ndarray = None,
        pin_format : bool = False) -> None:
        
        # Reset to keyframe
        if q is None and v is None:
            mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)

        v_mj = np.zeros(self.mj_model.nv) if v is None else v
        if pin_format: q_mj, v_mj = pin_2_mj_qv(q, v_mj)
        self._set_state(q_mj, v_mj)

    def _set_state(self, q_mj : np.ndarray, v_mj : np.ndarray):
        self.mj_data.qpos = q_mj.copy()
        self.mj_data.qvel = v_mj.copy()
        self.mj_data.ctrl = np.zeros(self.mj_model.nu)
        mujoco.mj_forward(self.mj_model, self.mj_data)       

    def setup(
        self,
        sim_time : float,
        viewer : bool,
        controller : Controller,
        data_recorder : DataRecorder,
        visual_callback : VisualCallback,
        ) -> None:

        self.controller = controller
        self.data_recorder = data_recorder
        self.visual_callback = visual_callback
        
        # Variables
        self.sim_step : int = 0
        self.viewer_step : int = 0
        self.time : float = 0.
        self.sim_time = sim_time

        # Threading for physics and viewer
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data) if viewer else None
        self.stop_sim = False

        if self.data_recorder: data_recorder.reset()

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

        # Record data
        if self.data_recorder is not None:
            self._record_data_step()
        
        # Apply force
        mujoco.mj_step2(self.mj_model, self.mj_data)
        
        self.sim_step += 1
        self.time += self.sim_dt

    def _run_physics(self):
        while not(self.stop_sim):
            self.locker.acquire()
            physics_time = self._physics_step()
            self.locker.release()

            self._stop_sim()

            if self.viewer:
                sleep_time = self.sim_dt - physics_time
                if sleep_time > 0.:
                    time.sleep(sleep_time)

        mujoco.mj_forward(self.mj_model, self.mj_data)       

    def _viewer_step(self) -> float:
        t_start = time.time()
        self.viewer.sync()
        self.viewer_step += 1
        render_time = (time.time() - t_start)

        return render_time

    def _run_viewer(self):
        while self.viewer.is_running() and not(self.stop_sim):
            self.locker.acquire()
            render_time = self._viewer_step()
            self._update_visual()
            self.locker.release()

            sleep_time = self.viewer_dt - render_time
            if sleep_time > 0.:
                time.sleep(sleep_time)

        self.viewer.close()
        self.stop_sim = True
        self.viewer = None

    def _stop_sim(self) -> None:
        if self.sim_step * self.sim_dt >= self.sim_time:
            self.stop_sim = True
        
        if self.controller and self.controller.diverged:
            self.stop_sim = True

    def _start_viewer(self,):
        viewer_thread = Thread(target=self._run_viewer)
        viewer_thread.start()

        return viewer_thread
    
    def _update_visual(self,):
        if (self.visual_callback and 
            self.viewer and
            self.viewer.is_running()):
            self.visual_callback.render(
                self.sim_step,
                self.viewer,
                self.mj_data
                )

    def run(self,
            sim_time : float = 5.,
            viewer : bool = True, 
            controller : Controller = None,
            data_recorder : DataRecorder = None,
            visual_callback : VisualCallback = None,
            ):
        
        self.setup(
            sim_time,
            viewer,
            controller,
            data_recorder,
            visual_callback,
            )

        # Start viewer and physics threads
        if viewer:
            viewer_thread = Thread(target=self._run_viewer)
            viewer_thread.start()
        else:
            viewer_thread = None
        
        physics_thread = Thread(target=self._run_physics)
        physics_thread.start()

        try:
            # Wait for threads to complete
            physics_thread.join()
            if viewer_thread: viewer_thread.join()

        except KeyboardInterrupt:
            print("\nSimulation interrupted.")
            self.stop_sim = True

            # Ensure threads are stopped gracefully
            if viewer_thread and viewer_thread.is_alive():
                viewer_thread.join()
            if physics_thread.is_alive():
                physics_thread.join()

        print("Simulation stopped.")

        if self.data_recorder:
            self.data_recorder.save()
