import numpy as np

DoF = 5  # Degrees of Freedom

def compute_exponential_and_J_s(S_list, theta_values):
    exp_matrix = np.eye(4)
    exp_current = np.eye(4)
    Adj = np.eye(6)
    J_s = np.zeros((6, DoF))

    for i in range(DoF):
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
    
    T_sb = np.matmul(exp_matrix, M)
    R = T_sb[:3, :3]
    p = T_sb[:3, 3]
    p_skew = np.array([[0, -p[2], p[1]],
                       [p[2], 0, -p[0]],
                       [-p[1], p[0], 0]])
    Adj = np.block([[R, np.zeros((3, 3))],
                    [np.matmul(p_skew, R), R]])
    
    return T_sb, J_s, Adj

def convert_J_s_into_J_b(J_s, Adj):
    J_b = np.matmul(np.linalg.inv(Adj), J_s)
    return J_b
    
def T_sb_to_twist(T_sb):
    # Prepare for Twist matrix
    T_bs = np.linalg.inv(T_sb)
    T_bd = np.matmul(T_bs, T_sd)
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

def IK(T_sd, M, S_list, initial_guess):
    theta_values = initial_guess.copy()
    T_sb, J_s, Adj = compute_exponential_and_J_s(S_list, theta_values)

    J_b = convert_J_s_into_J_b(J_s, Adj)
    J_b_pinv = np.linalg.pinv(J_b[3:,:])

    V_b, omega, v = T_sb_to_twist(T_sb)
    
    omega_threshold = 0.001
    v_threshold = 0.0001
    i = 0
    max_iterations = 100

    err = np.any(np.abs(v) > v_threshold)

    while err and i < max_iterations:
        # Compute delta theta
        delta_theta = np.matmul(J_b_pinv, V_b[3:,:]).flatten().tolist()
        # Update theta values
        theta_values = [theta_values[j] + delta_theta[j] for j in range(DoF)]
        # Computation of the current iteration
        T_sb, J_s, Adj = compute_exponential_and_J_s(S_list, theta_values)

        J_b = convert_J_s_into_J_b(J_s, Adj)
        J_b_pinv = np.linalg.pinv(J_b[3:,:])

        V_b, omega, v = T_sb_to_twist(T_sb)
        # Check for convergence
        err = np.any(np.abs(v) > v_threshold)
        i += 1

    if i >= max_iterations:
        converge = False
    else:
        converge = True    
    
    return theta_values, i, converge
    










if __name__ == "__main__":
    # M Matrix in Home Configuration
    M = np.array([[ 0 , 0 , 1, 42.5 ],
                  [-1 , 0 , 0, 0    ],
                  [ 0 ,-1 , 0, 30.4 ],
                  [ 0 , 0 , 0, 1    ]])


    # Screw Axes in Space form
    S_list = np.array([[0,  0   ,  0   ,  0    , 1   ],
                       [0, -1   , -1   , -1    , 0   ],
                       [1,  0   ,  0   ,  0    , 0   ], 
                       [0,  15.5,  30.4,  30.4 , 0   ],
                       [0,  0   ,  0   ,  0    , 30.4],
                       [0,  0   ,  0   , -13.9 , 0   ]]).T


    
    # Desired end effector configuration
    constraints = [2048, 2048, 2048, 2048, 4096]
    x,y,z = 17, 5, 2
    T_sd = np.array([[ 0,-1, 0, x],
                     [-1, 0, 0, y],
                     [ 0, 0,-1, z],
                     [ 0, 0, 0, 1]])
    
    random_ranges = [
        (-np.pi/2, np.pi/2),
        (-np.pi/2, np.pi/2),
        (0, 2*np.pi),
        (-np.pi/2, np.pi/2),
        (-np.pi/2, np.pi/2),
    ]
    # print("Random ranges 1 lowest value:", random_ranges[0][0])
    
    initial_guess = np.array([np.random.uniform(low, high) for low, high in random_ranges])
    converge = False
    within_range = False
    integer_values = np.zeros(DoF, dtype=int)
    trial = 0
    max_trial = 100

    while within_range == False and trial < max_trial:
        theta_sol, iterations, converge = IK(T_sd, M, S_list, initial_guess)
        # No convergence
        if converge == False:
            initial_guess = np.array([np.random.uniform(low, high) for low, high in random_ranges])
            trial += 1
            continue
        # Convergence
        for i in range(DoF):
            theta_sol[i] = theta_sol[i] % (2 * np.pi)
            theta_sol[i] = np.round(theta_sol[i],5)
            if theta_sol[i] > np.pi:
                theta_sol[i] -= 2 * np.pi

        # Check within range
        if all(random_ranges[i][0] <= theta_sol[i] <= random_ranges[i][1] for i in range(DoF)) == True:
            within_range = True
            trial += 1
            break
        initial_guess = np.array([np.random.uniform(low, high) for low, high in random_ranges])
        trial += 1

    print()
    print(f"Final Joint Angles (in radians): \n {theta_sol}")

    print()
    print(f"Final Joint Angles (in degrees): \n {[np.round(np.degrees(angle), 3) for angle in theta_sol]}")

    print()
    T_sb, J_s, Adj = compute_exponential_and_J_s(S_list, theta_sol)
    np.set_printoptions(suppress=True)
    print("Final Transformation Matrix:\n", np.round(T_sb,5))
    print()
    print("Convergence:", converge)
    print("Iterations:", iterations)
    print("Within range:", within_range)
    print("Trial:", trial)