#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from interfaces.srv import ComputeIK, ControlMotor  # Replace with your actual service interface
import threading

class UserNode(Node):
    def __init__(self):
        super().__init__("ik_client")
        self.IK_client = self.create_client(ComputeIK, "computation_service")
        # self.motor_client = self.create_client(ControlMotor, "control_motor")

    def call_compute_IK(self):
        while not self.IK_client.wait_for_service(1.0):
            self.get_logger().warn("Waiting for 'compute_IK' service to become available...")
        self.get_logger().info("'compute_IK' service is now available.")

        print()
        # Ask the user for the desired (x, y, z) position
        while True:            
            print("Please enter the desired position in the format x; y; z:")
            position_input = input()
            try:
                x, y, z = map(float, position_input.split(";"))
                if ((x-1.8)**2 + y**2 + (z-9.7)**2 > 35.3**2) or (z < 9.7):
                    self.get_logger().error("Position out of reach. Please enter a valid position (x-1.8)^2 + y^2 + (z-9.7)^2 <= 35.2^2 and z >= 9.7.")
                else:
                    break  # Exit the loop if the input is valid
            except ValueError:
                self.get_logger().error("Invalid input format. Please use the format (x; y; z).")

        print()
        # Ask the user for the initial joint angles in degrees
        joint_angles = [0.0] * 6 
        # for i in range(0, 6):
        #     while True:                                
        #         try:
        #             joint_angles[i] = float(input(f"Enter initial guess for joint {i+1} angle (in degrees): "))
        #             break
        #         except ValueError:
        #             self.get_logger().error(f"Invalid input for joint {i+1}. Please enter a valid number.")
        joint_angles = [20.0, 20.0, 20.0, 20.0, 20.0, 20.0]  # Example initial guess

        print()
        # Create a request object
        ik_request = ComputeIK.Request()
        ik_request.x = x
        ik_request.y = y
        ik_request.z = z
        ik_request.initial_guess_joint_angles = joint_angles
        
        print()
        # Send the request and handle the response asynchronously
        compute_IK_future = self.IK_client.call_async(ik_request)
        compute_IK_future.add_done_callback(self.callback_compute_IK_response)

    def callback_compute_IK_response(self, compute_IK_future):
        try:
            response = compute_IK_future.result()
            self.get_logger().info(f"Desired joint angles: {response.desired_joint_angles}")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
        
        print()
        # Send the joint angles to Node 3 (Robot Control) via the service
        # self.call_control_motor(response.desired_joint_angles)      

    def call_control_motor(self, joint_angles):
        while not self.motor_client.wait_for_service(1.0):
            self.get_logger().warn("Waiting for 'control_motor' service to become available...")
        self.get_logger().info("'control_motor' service is now available.")

        print()
        motor_request = ControlMotor.Request()
        motor_request.desired_joint_angles = joint_angles
        
        # Send the request and handle the response asynchronously
        control_motor_future = self.motor_client.call_async(motor_request)
        control_motor_future.add_done_callback(self.callback_control_motor_response)

    def callback_control_motor_response(self, control_motor_future):
        try:
            response = control_motor_future.result()
            self.get_logger().info(f"Final joint angles: {response.final_joint_angles}")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = UserNode()
    def input_loop():
            while rclpy.ok():
                node.call_compute_IK()

    # Run user input in background thread
    input_thread = threading.Thread(target=input_loop, daemon=True)
    input_thread.start()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()