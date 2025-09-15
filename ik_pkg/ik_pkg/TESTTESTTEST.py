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

        self.DoF = 5  # Degrees of Freedom
        self.T_sd = np.eye(4)  # Desired end effector configuration, to be set in the callback_compute method

    def callback_compute(self, request: ComputeIK.Request, response: ComputeIK.Response):
        # Desired end effector configuration
        self.T_sd = np.array([[ 0, -1, 0, request.x],
                         [ -1, 0, 0, request.y],
                         [ 0, 0, -1, request.z],
                         [ 0, 0, 0, 1]])
        
        random_ranges = [
            (-0.68, 0.68),
            (-0.73, 0),
            (0, 0.73),
            (-np.pi/2, -1),
            (0, 2*np.pi)
        ]

        initial_guess = np.array([np.random.uniform(low, high) for low, high in random_ranges])
        converge = False
        within_range = False

        theta_sol, success, i = self.inverse_kinematics_space(self.s_list, self.M, self.T_sd, initial_guess, max_iters=100)

        # print(theta_sol)
        theta_sol = theta_sol.astype(np.float32)
        response.desired_joint_angles = theta_sol
        response.converged = success

        print("Result:", theta_sol)
        print("Converged:", success, "after", i, "iterations")
        print("RESULT POSE:")
        print(self.fk_poe(self.s_list, theta_sol, self.M))

        return response

    def inverse_kinematics_space(self, xi_list, gst0, g_desired, theta_init=None, tol=1e-4, max_iters=100):
        n = len(xi_list)
        if theta_init is None:
            theta = np.zeros(n)
        else:
            theta = np.array(theta_init, dtype=float)
        e_omega = 1e-3  # angular velocity threshold
        e_v = 1e-4      # linear velocity threshold
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
    
    def compute_exponential_and_J_s(self, S_list, theta_values):
        exp_matrix = np.eye(4)
        exp_current = np.eye(4)
        Adj = np.eye(6)
        J_s = np.zeros((6, self.DoF))

        for i in range(self.DoF):
            S = S_list[i]
            theta = theta_values[i]
            J_s[:, i] = np.matmul(Adj, S)

            # Rotation and translation
            omega = S[:3]
            v = S[3:]
            omega_skew = np.array([[0, -omega[2], omega[1]],
                                [omega[2], 0, -omega[0]],
                                [-omega[1], omega[0], 0]])
            R = np.eye(3) + np.sin(theta) * omega_skew + (1 - np.cos(theta)) * np.matmul(omega_skew, omega_skew)
            p = (np.eye(3) * theta + (1 - np.cos(theta)) * omega_skew + (theta - np.sin(theta)) * np.matmul(omega_skew, omega_skew)).dot(v)
            exp_current = np.block([[R, p.reshape(3, 1)],
                                    [0, 0, 0, 1]])
                
            exp_matrix = np.matmul(exp_matrix, exp_current)
            # Update the adjoint matrix
            R = exp_matrix[:3, :3]
            p = exp_matrix[:3, 3]
            p_skew = np.array([[0, -p[2], p[1]],
                            [p[2], 0, -p[0]],
                            [-p[1], p[0], 0]])
            Adj = np.block([[R, np.zeros((3, 3))],
                        [np.matmul(p_skew, R), R]])
        
        T_sb = np.matmul(exp_matrix, self.M)
        R = T_sb[:3, :3]
        p = T_sb[:3, 3]
        p_skew = np.array([[0, -p[2], p[1]],
                        [p[2], 0, -p[0]],
                        [-p[1], p[0], 0]])
        Adj = np.block([[R, np.zeros((3, 3))],
                        [np.matmul(p_skew, R), R]])
        
        return T_sb, J_s, Adj

    def convert_J_s_into_J_b(self, J_s, Adj):
        J_b = np.matmul(np.linalg.inv(Adj), J_s)
        return J_b
    
    def T_sb_to_twist(self, T_sb):
        # Prepare for Twist matrix
        T_bs = np.linalg.inv(T_sb)
        T_bd = np.matmul(T_bs, self.T_sd)
        R_bd = T_bd[:3, :3]
        p_bd = T_bd[:3, 3]

        # Angular velocity omega
        cos_theta = (np.trace(R_bd) - 1) / 2
        cos_theta = np.clip(cos_theta, -1, 1)  # Ensure cos_theta is within valid range
        theta = np.arccos(cos_theta)

        if cos_theta >= 1:
            omega = [0, 0, 0]

        elif cos_theta <= -1:
            if not 1 + R_bd[2][2] <= 1e-6:
                omega = (theta / np.sqrt(2 * (1 + R_bd[2][2]))) * np.array([R_bd[0][2], R_bd[1][2], 1 + R_bd[2][2]])
            elif not 1 + R_bd[1][1] <= 1e-6:
                omega = (theta / np.sqrt(2 * (1 + R_bd[1][1]))) * np.array([R_bd[0][1], 1 + R_bd[1][1], R_bd[2][1]])
            else:
                omega = (theta / np.sqrt(2 * (1 + R_bd[0][0]))) * np.array([1 + R_bd[0][0], R_bd[1][0], R_bd[2][0]])

        else:
            omega = (theta / (2 * np.sin(theta))) * np.array([R_bd[2][1] - R_bd[1][2], R_bd[0][2] - R_bd[2][0], R_bd[1][0] - R_bd[0][1]])

        omega_skew = np.array([[0, -omega[2], omega[1]],
                            [omega[2], 0, -omega[0]],
                            [-omega[1], omega[0], 0]])

        # Linear velocity v
        if cos_theta >= 1:
            v = p_bd
        else:    
            G_inv = np.eye(3) - (0.5 * omega_skew) + (((1.0 / theta - 0.5 * (1 / np.tan(theta / 2.0))) * np.dot(omega_skew, omega_skew)) / theta)
            v = np.matmul(G_inv, p_bd)

        # Twist V_b
        V_b = np.zeros((6, 1))
        omega = np.array(omega, dtype=float)
        v = np.array(v, dtype=float)
        V_b[0:3, 0] = omega
        V_b[3:6, 0] = v

        return V_b, omega, v    

    def IK(self, T_sd, M, S_list, initial_guess):
        theta_values = initial_guess.copy()
        T_sb, J_s, Adj = self.compute_exponential_and_J_s(S_list, theta_values)

        J_b = self.convert_J_s_into_J_b(J_s, Adj)
        J_b_pinv = np.linalg.pinv(J_b[3:,:])

        V_b, omega, v = self.T_sb_to_twist(T_sb)
        
        omega_threshold = 0.001
        v_threshold = 0.0001
        i = 0
        max_iterations = 100

        err = np.any(np.abs(v) > v_threshold)

        while err and i < max_iterations:
            # Compute delta theta
            delta_theta = np.matmul(J_b_pinv, V_b[3:,:]).flatten().tolist()
            # Update theta values
            theta_values = [theta_values[j] + delta_theta[j] for j in range(self.DoF)]
            # Computation of the current iteration
            T_sb, J_s, Adj = self.compute_exponential_and_J_s(S_list, theta_values)

            J_b = self.convert_J_s_into_J_b(J_s, Adj)
            J_b_pinv = np.linalg.pinv(J_b[3:,:])

            V_b, omega, v = self.T_sb_to_twist(T_sb)
            # Check for convergence
            err = np.any(np.abs(v) > v_threshold)
            i += 1

        if i >= max_iterations:
            converge = False
        else:
            converge = True    
        
        return theta_values, i, converge

def main(args=None):
    rclpy.init(args=args)
    node = KinematicsNode() # MODIFY NAME
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()