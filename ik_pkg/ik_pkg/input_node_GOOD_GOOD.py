#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from interfaces.srv import ComputeIK, ControlMotor  # Replace with your actual service interface
import threading

class UserNode(Node):

    guess = [0.0] * 6

    def __init__(self):
        super().__init__("ik_client")
        self.IK_client = self.create_client(ComputeIK, "computation_service")
        # self.motor_client = self.create_client(ControlMotor, "control_motor")
        self.ready_for_next = threading.Event()
        self.ready_for_next.set()  # Allow first run

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

        print()
        # Create a request object
        ik_request = ComputeIK.Request()
        ik_request.x = x
        ik_request.y = y
        ik_request.z = z
        ik_request.initial_guess_joint_angles = UserNode.guess
        
        print("\n initial guess joint angles \n", ik_request.initial_guess_joint_angles)

        print()
        # Send the request and handle the response asynchronously
        compute_IK_future = self.IK_client.call_async(ik_request)
        compute_IK_future.add_done_callback(self.callback_compute_IK_response)

    def callback_compute_IK_response(self, compute_IK_future):
        try:
            response = compute_IK_future.result()
            self.get_logger().info(f"Desired joint angles: {response.desired_joint_angles}")
            UserNode.guess = response.desired_joint_angles

            print("\n result: \n",UserNode.guess)
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
        finally:
            self.ready_for_next.set()  # Allow next round   

        print()


def main(args=None):
    rclpy.init(args=args)
    node = UserNode()

    def input_loop():
        while rclpy.ok():
            node.ready_for_next.wait()     # Wait until previous call is done
            node.ready_for_next.clear()    # Lock until next response completes
            node.call_compute_IK()

    input_thread = threading.Thread(target=input_loop, daemon=True)
    input_thread.start()

    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()