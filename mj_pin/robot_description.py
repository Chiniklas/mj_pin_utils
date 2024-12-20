from typing import Any
from mj_pin.abstract import RobotDescription, QuadrupedDescription

Go2Description = QuadrupedDescription(
    name = "go2",
    eeff_frame_name = ["FL", "FR", "RL", "RR"],
    foot_size = 0.012,
)

class RobotDescriptionFactory:
    AVAILABLE_ROBOT = {
        Go2Description.name : Go2Description,
    }

    @staticmethod
    def get(robot_name : str) -> RobotDescription | Any:
        description = RobotDescriptionFactory.AVAILABLE_ROBOT.get(robot_name, RobotDescription(robot_name))
        
        return description
