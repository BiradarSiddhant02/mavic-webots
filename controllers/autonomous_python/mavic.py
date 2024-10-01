from controller import Robot, Camera, GPS, InertialUnit, Motor, LED
from typing import Tuple

from constants import (K_VERTICAL_THRUST, K_VERTICAL_OFFSET,
                       K_VERTICAL_P, K_VERTICAL_I, K_VERTICAL_D,
                       K_ROLL_P, K_PITCH_P, MAX_SIMULATION_TIME, MAX_MOTOR_VELOCITY)

class Mavic:
    def __init__(self):
        
        ## initialize Robot
        self.drone = Robot()
        self.timestep = int(self.drone.getBasicTimeStep())

        ## Get Camera
        self.camera = self.drone.getDevice("camera")
        self.camera.enable(self.timestep)

        ## Get IMU
        self.imu = self.drone.getDevice("inertial unit")
        self.imu.enable(self.timestep)

        ## Get gyro
        self.gyro = self.drone.getDevice("gyro")
        self.gyro.enable(self.timestep)
        
        ## Get GPS
        self.gps = self.drone.getDevice("gps")
        self.gps.enable(self.timestep)

        ## Get Propellers
        self.front_left_motor = self.drone.getDevice("front left propeller")
        self.front_right_motor = self.drone.getDevice("front right propeller")
        self.rear_left_motor = self.drone.getDevice("rear left propeller")
        self.rear_right_motor = self.drone.getDevice("rear right propeller")

        self.motors = [
            self.front_left_motor,
            self.front_right_motor,
            self.rear_left_motor,
            self.rear_right_motor
        ]

        for motor in self.motors:
            motor.setPosition(float('inf'))  # Set motors to velocity mode
            motor.setVelocity(1.0)  # Initial velocity for motors

    def get_imu_values(self) -> Tuple[float, float, float]:
        return self.imu.getRollPitchYaw()
    
    def get_gps_values(self) -> Tuple[float, float, float]:
        return self.gps.getValues()
    
    def get_gyro_values(self) -> Tuple[float, float, float]:
        return self.gyro.getValues()
    
    def get_time(self) -> float:
        return self.drone.getTime()
    
    def set_rotor_speed(self, speed: Tuple[float, float, float, float]) -> None:
        fl_speed, fr_speed, rl_speed, rr_speed = speed

        self.front_left_motor.setVelocity(fl_speed)
        self.front_right_motor.setVelocity(fr_speed)
        self.rear_left_motor.setVelocity(rl_speed)
        self.rear_right_motor.setVelocity(rr_speed)

