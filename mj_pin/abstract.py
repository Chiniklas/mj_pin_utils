import time
import mujoco
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import wraps
from datetime import datetime
import multiprocessing as mp
from .ext.keyboard import KBHit

class Colors():
    RED =     (1.0, 0.0, 0.0, 1.)
    GREEN =   (0.0, 1.0, 0.0, 1.)
    BLUE =    (0.0, 0.0, 1.0, 1.)
    YELLOW =  (1.0, 1.0, 0.0, 1.)
    WHITE =   (0.9, 0.9, 0.9, 1.)
    BLACK =   (0.1, 0.1, 0.1, 1.)

    COLOR_ID_MAP = {
            0 : RED,
            1 : GREEN,
            2 : BLUE,
            3 : YELLOW,
            4 : WHITE,
            5 : BLACK,
        }

    COLOR_NAME_MAP = {
        "red" : RED,
        "green" : GREEN,
        "blue" : BLUE,
        "yellow" : YELLOW,
        "white" : WHITE,
        "black" : BLACK,
    }

    @staticmethod
    def id(id : int) -> List[str]:
        return dict.get(Colors.COLOR_ID_MAP, id, Colors.WHITE)
    
    @staticmethod
    def name(name : str) -> List[str]:
        return dict.get(Colors.COLOR_NAME_MAP, name, Colors.WHITE)

def call_every(func):
    """
    Decorator to call the decorated function only every `self.call_every` steps.
    """
    @wraps(func)
    def wrapper(self, sim_step: int, *args, **kwargs):
        if sim_step % self._call_every == 0:
            return func(self, sim_step, *args, **kwargs)
    return wrapper

class Controller(ABC):
    def __init__(self) -> None:
        # Stop simulation if diverged
        self.diverged : bool = False

    def get_state(self, mj_data) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get state in mujoco format from mujoco data.
        q : [
             x, y, z,
             qw, qx, qy, qz,
             j1, ..., je,
            ]
        
        v : [
             vx, vy, vz, (global frame)
             wx, wy, wz, (local frame)
             vj1, ..., vje,
            ]

        Returns:
            Tuple[np.ndarray, np.ndarray]: q [nq], v [nv]
        """
        q = mj_data.qpos.copy()
        v = mj_data.qvel.copy()

        return q, v
        
    @abstractmethod
    def get_torques(
        self,
        sim_step : int,
        mj_data,
    ) -> Dict[str, float]:
        pass

class Keyboard(ABC):
    KEYBOARD_UPDATE_FREQ = 50

    def __init__(self) -> None:
        super().__init__()

        self.keyboard = KBHit()
        self.last_key: str = ""
        self.stop = False
        self.update_thread = None
        self._start_update_thread()

    def _keyboard_thread(self):
        """
        Update base goal location based on keyboard events.
        """
        while not self.stop:
            if self.last_key == '\n':  # ENTER
                break
            if self.keyboard.kbhit():
                self.last_key = self.keyboard.getch()
            else:
                self.last_key = ""

            self.on_key()
            time.sleep(1. / Keyboard.KEYBOARD_UPDATE_FREQ)

    def _start_update_thread(self):
        if self.update_thread is None or not self.update_thread.is_alive():
            self.update_thread = threading.Thread(target=self._keyboard_thread)
            self.update_thread.start()

    def _stop_update_thread(self):
        self.last_key = '\n'
        self.stop = True
        if self.update_thread is not None and self.update_thread.is_alive():
            self.update_thread.join()
        self.update_thread = None

    def __del__(self):
        self._stop_update_thread()

    @abstractmethod
    def on_key(self, **kwargs) -> str:
        pass

class DataRecorder(ABC):
    def __init__(
        self,
        record_dir : str = "",
        record_step : int = 1,
        ) -> None:
        self.record_dir = record_dir
        self._call_every = record_step

    @staticmethod
    def get_date_time_str() -> str:
        now = datetime.now()
        date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
        return date_time

    @call_every
    def _record(
        self,
        sim_step : int,
        mj_data,
        **kwargs,
    ) -> None:
        self.record(mj_data, **kwargs)
        
    @abstractmethod
    def record(
        self,
        mj_data,
        **kwargs,
    ) -> None:
        pass
    
    @abstractmethod
    def reset(self) -> None:
        pass
    
    @abstractmethod
    def save(self) -> None:
        pass

class VisualCallback(ABC):

    def __init__(self, update_step: int = 1):
        """
        Abstract base class for a MuJoCo viewer callback.

        Args:
            update_step (int): Number of simulation steps between each call to `render`.
        """
        super().__init__()
        self._call_every = update_step
        self.i_geom: int = 0
        self._geom_args = {}
        self.colors = Colors()

    def _add_geom(self, geom_type, pos, rot, size, rgba):
        """
        Add a geometry to the viewer's scene.

        Args:
            viewer: MuJoCo viewer instance.
            geom_type: Geometry type (e.g., `mujoco.mjtGeom.mjGEOM_SPHERE`).
            pos: Position of the geometry in world coordinates.
            rot: Rotation matrix (3x3) as a flattened array.
            size: Size of the geometry (array-like).
            rgba: RGBA rgba of the geometry.
        """
        self._geom_args[self.i_geom] = [
            geom_type,
            size,
            pos,
            rot.flatten(),
            rgba,
        ]
        self.i_geom += 1

    def add_sphere(self, pos, radius, rgba):
        """
        Add a sphere to the viewer's scene.

        Args:
            viewer: MuJoCo viewer instance.
            pos: Position of the sphere in world coordinates.
            size: Radius of the sphere.
            rgba: RGBA rgba of the sphere.
        """
        self._add_geom(
            geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=pos,
            rot=np.eye(3),
            size=[radius, 0, 0],
            rgba=rgba,
        )

    def add_box(self, pos, rot, size, rgba):
        """
        Add a box to the viewer's scene.

        Args:
            viewer: MuJoCo viewer instance.
            pos: Position of the box in world coordinates.
            rot: Rotation matrix (3x3) as a flattened array.
            size: Dimensions of the box (length, width, height).
            rgba: RGBA rgba of the box.
        """
        self._add_geom(
            geom_type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos,
            rot=rot,
            size=size,
            rgba=rgba,
        )

    @call_every
    def render(self, sim_step, viewer, mj_data):
        """
        Render the scene by calling `_render`.

        Args:
            viewer: MuJoCo viewer instance.
            sim_step: Current simulation step.
            mj_data: MuJoCo data instance.
        """
        self.i_geom = 0
        self.add_visuals(mj_data)

        for i, geom_args in self._geom_args.items():
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[i],
                *geom_args
            )

        viewer.user_scn.ngeom = self.i_geom

    @abstractmethod
    def add_visuals(self, viewer, sim_step, mj_data):
        """
        Abstract method to define rendering logic.

        Args:
            viewer: MuJoCo viewer instance.
            sim_step: Current simulation step.
            mj_data: MuJoCo data instance.
        """
        pass

class ParallelSimulatorBase(ABC):
    
    def __init__(self, n_cores: int = 1):
        self.n_cores = n_cores
        self.job_queue = mp.Queue(maxsize=n_cores+1)
        self.stop_processes = mp.Value('b', False)
        self.job_id = 0
        self.processes = []
        
        self.queue_timeout = 5.
        self.waiting_sleep_time = .1
        
    @abstractmethod
    def create_job(self) -> dict:
        """
        Producer.
        Creates arguments for the run_job method.
        """
        pass
    
    @abstractmethod
    def run_job(self, **kwargs):
        """
        Consumer.
        Processes the job with the arguments provided.
        """
        pass
    
    def _add_job(self) -> None:
        """
        Add a job to the workers queue.
        """
        while self.job_queue.full():
            time.sleep(self.waiting_sleep_time)
            
        try:
            kwargs = self.create_job()  # Get job arguments
        except Exception as e:
            print(e)
            return
            
        self.job_queue.put((self.job_id, kwargs))
        print("Added job", self.job_id)
        self.job_id += 1
    
    def _run_job(self, worker_id: int) -> None:
        """
        Worker method that runs jobs from the queue.
        """       
        while not self.stop_processes.value:
            try:
                job_id, kwargs = self.job_queue.get(block=True, timeout=self.waiting_sleep_time)
                
                print(f"Job {job_id} running (worker {worker_id}).")
                self.run_job(**kwargs)
                print(f"Job {job_id} done (worker {worker_id}).")

            except mp.queues.Empty:
                continue
            
            except Exception as e:
                print(f"Worker {worker_id} encountered an error: {e}")
                break

        print(f"Worker {worker_id} stopping...")

    def run(self, N: int):
        """
        Run N iterations of the producer-consumer process in parallel.
        """
        # Start worker processes
        self.processes = []
        for i in range(self.n_cores):
            p = mp.Process(target=self._run_job, args=(i,))
            p.start()
            self.processes.append(p)

        # Produce N jobs
        for _ in range(N):
            self._add_job()

        # Wait until the queue is empty
        while self.job_queue.qsize() > 0:
            time.sleep(self.queue_timeout)

        # Stop all processes
        self.stop()

    def stop(self):
        """
        Stop all worker processes.
        """
        self.stop_processes.value = True
        if self.processes:
            print("Stopping all processes...")
            for i, p in enumerate(self.processes):
                p.join()
                print(f"Worker {i} stopped.")
        self.processes = []

    def __del__(self):
        self.stop()

@dataclass
class RobotDescription(ABC):
    # Robot name
    name : str
    # MuJoCo model path (id loaded)
    xml_path : str = ""
    # Pinocchio model path (if loaded)
    urdf_path : str = ""
    # Scene path 
    xml_scene_path : str = ""
    # End-effectors frame name
    eeff_frame_name : Optional[List[str]] = None
    # Nominal configuration
    q0 : Optional[np.ndarray] = None