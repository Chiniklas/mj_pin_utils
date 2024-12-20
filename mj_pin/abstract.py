import mujoco
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import wraps
from datetime import datetime

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
    def record(
        self,
        sim_step : int,
        mj_data,
        **kwargs,
    ) -> None:
        self._record(mj_data, **kwargs)
        
    @abstractmethod
    def _record(
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

    RED =     [1.0, 0.0, 0.0, 1.]
    GREEN =   [0.0, 1.0, 0.0, 1.]
    BLUE =    [0.0, 0.0, 1.0, 1.]
    YELLOW =  [1.0, 1.0, 0.0, 1.]
    WHITE =   [0.9, 0.9, 0.9, 1.]
    BLACK =   [0.1, 0.1, 0.1, 1.]
    
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

        self.colors_id = {
            0 : VisualCallback.RED,
            1 : VisualCallback.GREEN,
            2 : VisualCallback.BLUE,
            3 : VisualCallback.YELLOW,
            4 : VisualCallback.WHITE,
            5 : VisualCallback.BLACK,
        }

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
        self._add_visuals(mj_data)

        for i, geom_args in self._geom_args.items():
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[i],
                *geom_args
            )

        viewer.user_scn.ngeom = self.i_geom

    @abstractmethod
    def _add_visuals(self, viewer, sim_step, mj_data):
        """
        Abstract method to define rendering logic.

        Args:
            viewer: MuJoCo viewer instance.
            sim_step: Current simulation step.
            mj_data: MuJoCo data instance.
        """
        pass
        
@dataclass
class RobotDescription(ABC):
    # Robot name
    name : str
    # End-effectors frame id
    eeff_frame_name : List[str] = None

    def __post_init__(self):
        # MuJoCo model path
        self.mjcf_path : str = ""
        # Pinocchio model path (if loaded)
        self.urdf_path : str = ""
        # Scene path
        self.scene_path : str = ""
        # Nominal configuration
        self.q0 : np.ndarray = None 

@dataclass
class QuadrupedDescription(RobotDescription):
    # Foot size
    foot_size : float = 0.