#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from scipy.linalg import expm, logm
from interfaces.srv import ComputeIK


class KinematicsNode(Node): # MODIFY NAME
    def __init__(self):
        super().__init__("kinematics_node") # MODIFY NAME
        self.computation_service_ = self.create_service(ComputeIK, "computation_service", self.callback_compute)

        self.get_logger().info("Kinematics Node started")

        self.M = np.array([[0,0,1,21.09],
                    [-1,0,0,0],
                    [0,-1,0,31.6],
                    [0,0,0,1]])
        
        self.s_list = np.array([[0, 0   , 0     , 1   , 0     , 1   ],
                        [0, -1  , -1    , 0   , -1    , 0   ],
                        [1, 0   ,  0    , 0   , 0     , 0   ], 
                        [0, 16.5, 31.6  , 0   , 31.6  , 0   ],
                        [0, 0   , 0     , 31.6, 0     , 31.6],
                        [0, 0   , 0     , 0   , -13.89, 0   ]]).T 
    
    def callback_compute(self, request: ComputeIK.Request, response: ComputeIK.Response):
        # theta_1 = [np.pi/4, -np.pi/4, np.pi/4, -np.pi/4, -np.pi/4, np.pi/4]
        if request.initial_guess_joint_angles is None:
            theta_0 = [0,0,0,0,0,0]
        else:
            theta_0 = request.initial_guess_joint_angles

        
        T_sd = np.array([[ 0, -1, 0, request.x],
                         [ -1, 0, 0, request.y],
                         [ 0, 0, -1, request.z],
                         [ 0, 0, 0, 1]])
        
        random_ranges = [
            (-np.pi/2, np.pi/2),
            (-np.pi/2, np.pi/2),
            (-np.pi/2, np.pi/2),
            (-np.pi/2, np.pi/2),
            (0, 2*np.pi),
            (-np.pi/2, np.pi/2),
        ]

        within_range = False
        trial = 0
        max_trial = 50
        while not within_range and trial < max_trial:
            # Call the inverse kinematics function
            theta_sol, success, i = self.inverse_kinematics_space(self.s_list, self.M, T_sd, theta_0, max_iters=50)

            if success == True:
                # print(theta_sol)
                #check if the solution is in the range
                for i in range(len(theta_sol)):
                    theta_sol[i] = theta_sol[i] % (2 * np.pi)
                    theta_sol[i] = np.round(theta_sol[i], 5)

                print("theta_sol:", theta_sol)
                if all(random_ranges[i][0] <= theta_sol[i] <= random_ranges[i][1] for i in range(len(theta_sol))):
                    print("Solution is within range")
                    print("Solution:", theta_sol)
                    within_range = True
                    trial += 1
                    break
                else:
                    theta_0 = np.array([np.random.uniform(low, high) for low, high in random_ranges])
                    trial += 1
                    continue
            else:
                theta_0 = np.array([np.random.uniform(low, high) for low, high in random_ranges])
                trial += 1
                continue

        # print(theta_sol)
        theta_sol = theta_sol.astype(np.float32)
        response.desired_joint_angles = theta_sol
        response.converged = success

        print("Result:", theta_sol)
        print("Converged:", success, "after", i, "iterations")
        print("RESULT POSE:")
        print(self.fk_poe(self.s_list, theta_sol, self.M))
        # self.get_logger().info("Result: ", theta_sol)
        # self.get_logger().info ("Converged: ", success)
        return response

    def inverse_kinematics_space(self, xi_list, gst0, g_desired, theta_init=None, tol=1e-4, max_iters=100):
        n = len(xi_list)
        if theta_init is None:
            theta = np.zeros(n)
        else:
            theta = np.array(theta_init, dtype=float)
        e_omega = 1e-1  # angular velocity threshold
        e_v = 1e-3      # linear velocity threshold
        for i in range(max_iters):
            g_current = self.fk_poe(xi_list, theta, gst0)
            Vs = self.calculate_twist_error_space(g_current, g_desired)
            omega = Vs[:3]
            v = Vs[3:]
            if np.linalg.norm(omega) < e_omega and np.linalg.norm(v) < e_v:
            # if np.linalg.norm(v) < e_v:
                theta = np.where(np.abs(theta) < 1e-6, 0, theta)
                # print("rounded theta:", theta)

                return theta, True, i
            Js = self.space_jacobian(xi_list, theta)
            dtheta = np.dot(np.linalg.pinv(Js), Vs)
            theta += dtheta

        theta = np.where(np.abs(theta) < 1e-6, 0, theta)
        # print("rounded theta:", theta)
        return theta, False, i  # If it didn't converge


    def near_zero(self, z):
        """
        Determines whether a scalar is near zero.
        :param z: scalar

        :return: bool
        """
        return abs(z) < 1e-6

    def skew(self, v):
        """
        Computes the skew-symmetric matrix of a 3D vector.
        :param v: 3D vector

        :return: 3x3 skew-symmetric matrix
        """
        result_skew = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
        return result_skew

    def twist_matrix(self, xi):
        """
        Computes the twist matrix from a twist vector.
        :param xi: 6D twist vector

        :return: 4x4 twist matrix
        """
        w = xi[:3]
        v = xi[3:]
        result_twist_matrix = np.block([[self.skew(w), v.reshape(3, 1)],
                     [0, 0, 0, 0]])
        return result_twist_matrix

    def exp_twist_matrix(self, twist_mat, theta):
        """
        :param twist_mat: 4x4 twist matrix
        :param theta: rotation angle (scalar)

        :return: 4x4 homogeneous transformation matrix
        """
        return expm(twist_mat * theta)

    def fk_poe(self, twist_list, theta_list, M):
        T = np.eye(4)
        for xi, theta in zip(twist_list, theta_list):
            T = T @ self.exp_twist_matrix(self.twist_matrix(xi), theta)

        T = T @ M
        return np.round(np.where(self.near_zero(T), 0, T), 4)

    def fk_poeB(self, twist_list, theta_list, M):
        T = np.eye(4)
        for xi, theta in zip(twist_list, theta_list):
            T = T @ self.exp_twist_matrix(self.twist_matrix(xi), theta)

        T = M @ T
        return np.round(np.where(self.near_zero(T), 0, T), 4)

    def se3_to_vec(self, se3mat):
        omega_hat = se3mat[0:3, 0:3]
        v = se3mat[0:3, 3]
        omega = np.array([omega_hat[2,1], omega_hat[0,2], omega_hat[1,0]])
        return np.concatenate((omega, v))

    def matrix_log6(self, T):
        return logm(T)

    def adjoint(self, T):
        """
        Computes the adjoint representation of a transformation matrix, for use in Inverse Kinematics.
        :param T: 4x4 transformation matrix

        :return: 6x6 adjoint matrix
        """
        R = T[0:3, 0:3]
        p = T[0:3, 3]
        p_skew = self.skew(p)
        Ad = np.zeros((6, 6))
        Ad[0:3, 0:3] = R
        Ad[3:6, 0:3] = p_skew @ R
        Ad[3:6, 3:6] = R
        return Ad

    def space_jacobian(self, xi_list, theta_list):
        """
        Computes the space Jacobian matrix for a given set of twists and joint angles.
        :param xi_list: list of twist vectors (6D)
        :param theta_list: list of joint angles (scalars)
        
        :return: 6xN Jacobian matrix
        """
        n = len(xi_list)
        Js = np.zeros((6, n))
        T = np.eye(4)
        for i in range(n):
            if i == 0:
                Js[:, 0] = xi_list[0]
            else:
                # Compute transformation up to joint i-1
                T = T @ expm(self.twist_matrix(xi_list[i-1]) * theta_list[i-1])
                Js[:, i] = self.adjoint(T) @ xi_list[i]
        return Js

    def calculate_twist_error_space(self, g_current, g_desired):
        g_error = g_desired @ np.linalg.inv(g_current)  # error in the SPACE frame!
        log_se3 = self.matrix_log6(g_error)
        Vs = self.se3_to_vec(log_se3)
        return Vs


def main(args=None):
    rclpy.init(args=args)
    node = KinematicsNode() # MODIFY NAME
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()