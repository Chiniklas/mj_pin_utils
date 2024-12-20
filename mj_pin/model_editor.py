import mujoco
import numpy as np
from typing import List

class ModelEditor():
    def __init__(self, mj_model):

        self.mj_model = mj_model
        self.mj_spec = mujoco.MjSpec()
        self.compiled : bool = True

    def add_geom(
        self,
        type,
        pos : np.ndarray,
        size : np.ndarray,
        rot : np.ndarray,
        color : List[float],
        ) -> int:
        pass

    def add_box(
        self,
        pos : np.ndarray,
        size : np.ndarray,
        rot : np.ndarray,
        color : List[float],
        ) -> int:
        pass

    def remove_geom(self, geom_id) -> None:
        pass

    def color_geom(self, geom_id) -> None:
        pass

    def move_geom(
        self, 
        geom_id, 
        new_pos, 
        new_rot
        ) -> None:
        pass
        
    def _init_spec(self):
        pass

    def _compile_spec(self):
        pass

    def get_model(self):
        pass