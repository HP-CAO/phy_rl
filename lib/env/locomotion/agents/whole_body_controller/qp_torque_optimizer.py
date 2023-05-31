"""Set up the zeroth-order QP problem for stance leg control.

For details, please refer to this paper:
https://arxiv.org/abs/2009.10019
"""

import numpy as np
import quadprog  # pytype:disable=import-error

np.set_printoptions(precision=3, suppress=True)

# ACC_WEIGHT = np.array([1., 1., 1., 10., 10, 1.])
# ACC_WEIGHT = np.array([1.75007132, 1.75113120, 1.79259481, 1.81761823, 1.81299985, 1.81365344])
ACC_WEIGHT = np.array([1.81666331, 1.81892955, 1.87235756, 1.89121405, 1.88585412, 1.88390268])


class QPTorqueOptimizer:
    """QP Torque Optimizer Class."""

    def __init__(self,
                 robot_mass,
                 robot_inertia,
                 friction_coef=0.45,
                 f_min_ratio=0.1,
                 f_max_ratio=10.):
        self.mpc_body_mass = robot_mass
        self.inv_mass = np.eye(3) / robot_mass
        self.inv_inertia = np.linalg.inv(robot_inertia.reshape((3, 3)))
        self.friction_coef = friction_coef
        self.f_min_ratio = f_min_ratio
        self.f_max_ratio = f_max_ratio

        # Precompute constraint matrix A
        self.A = np.zeros((24, 12))
        for leg_id in range(4):
            self.A[leg_id * 2, leg_id * 3 + 2] = 1
            self.A[leg_id * 2 + 1, leg_id * 3 + 2] = -1

        # Friction constraints
        for leg_id in range(4):
            row_id = 8 + leg_id * 4
            col_id = leg_id * 3
            self.A[row_id, col_id:col_id + 3] = np.array([1, 0, self.friction_coef])
            self.A[row_id + 1,
            col_id:col_id + 3] = np.array([-1, 0, self.friction_coef])
            self.A[row_id + 2,
            col_id:col_id + 3] = np.array([0, 1, self.friction_coef])
            self.A[row_id + 3,
            col_id:col_id + 3] = np.array([0, -1, self.friction_coef])

    def compute_mass_matrix(self, foot_positions):
        mass_mat = np.zeros((6, 12))
        mass_mat[:3] = np.concatenate([self.inv_mass] * 4, axis=1)

        for leg_id in range(4):
            x = foot_positions[leg_id]
            foot_position_skew = np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]],
                                           [-x[1], x[0], 0]])
            mass_mat[3:6, leg_id * 3:leg_id * 3 + 3] = self.inv_inertia.dot(foot_position_skew)
        return mass_mat

    def compute_constraint_matrix(self, contacts):
        f_min = self.f_min_ratio * self.mpc_body_mass * 9.8  # check the force bounds
        f_max = self.f_max_ratio * self.mpc_body_mass * 9.8
        lb = np.ones(24) * (-1e-7)
        contact_ids = np.nonzero(contacts)[0]
        lb[contact_ids * 2] = f_min
        lb[contact_ids * 2 + 1] = -f_max
        return self.A.T, lb

    def compute_objective_matrix(self, mass_matrix, desired_acc, acc_weight,
                                 reg_weight):
        g = np.array([0., 0., 9.8, 0., 0., 0.])
        Q = np.diag(acc_weight)  # todo replace this with our design
        # print(Q.shape)
        # Q = np.array([[1.85711137e-04, -1.03285919e-07, -5.27240018e-09, -1.45858161e-06,
        #                -4.16850481e-06, 5.81187834e-08, 1.14608179e-05, -5.52314267e-06,
        #                -2.61740507e-07, -6.49097633e-05, -3.04741153e-04, 5.01076079e-06, ],
        #               [-1.03285919e-07, 1.86190510e-04, 5.62956941e-09, 7.44926917e-06,
        #                1.75178044e-06, -1.06153105e-07, -2.19742182e-06, 6.54409552e-05,
        #                6.48743768e-08, 7.25879521e-04, 8.70794914e-05, -2.38874687e-05, ],
        #               [-5.27240018e-09, 5.62956941e-09, 1.85566582e-04, 8.36636556e-08,
        #                1.45746319e-07, -3.42594351e-09, -2.85423700e-07, 3.83933342e-07,
        #                3.33929395e-06, 4.57998775e-06, 1.06955724e-05, -2.58476322e-07, ],
        #               [-1.45858161e-06, 7.44926917e-06, 8.36636556e-08, 2.76839614e-04,
        #                2.60791981e-05, -1.17523305e-06, -3.68376504e-05, 7.77227489e-04,
        #                1.19100905e-06, 9.11386288e-03, 1.44813435e-03, -3.04261143e-04, ],
        #               [-4.16850481e-06, 1.75178044e-06, 1.45746319e-07, 2.60791981e-05,
        #                3.43156887e-04, -1.09406888e-06, -2.94763874e-04, 4.53104156e-05,
        #                9.44403776e-06, 5.30065166e-04, 1.15919271e-02, -1.21550736e-04, ],
        #               [5.81187834e-08, -1.06153105e-07, -3.42594351e-09, -1.17523305e-06,
        #                -1.09406888e-06, 1.85690502e-04, -3.13880068e-07, -6.38163946e-07,
        #                1.25357207e-09, -6.66190298e-06, 2.09566793e-05, 9.72255736e-06, ],
        #               [1.14608179e-05, -2.19742182e-06, -2.85423700e-07, -3.68376504e-05,
        #                -2.94763874e-04, -3.13880068e-07, 6.86289319e-04, -1.40682504e-04,
        #                -1.60556509e-05, -1.64727541e-03, -1.97004768e-02, 2.30538562e-04, ],
        #               [-5.52314267e-06, 6.54409552e-05, 3.83933342e-07, 7.77227489e-04,
        #                4.53104156e-05, -6.38163946e-07, -1.40682504e-04, 4.58766768e-03,
        #                4.56794970e-06, 5.16192201e-02, 5.52673548e-03, -1.69964228e-03, ],
        #               [-2.61740507e-07, 6.48743768e-08, 3.33929395e-06, 1.19100905e-06,
        #                9.44403776e-06, 1.25357207e-09, -1.60556509e-05, 4.56794970e-06,
        #                1.85425619e-04, 5.35067147e-05, 6.31241638e-04, -7.41514786e-06, ],
        #               [-6.49097633e-05, 7.25879521e-04, 4.57998775e-06, 9.11386288e-03,
        #                5.30065166e-04, -6.66190298e-06, -1.64727541e-03, 5.16192201e-02,
        #                5.35067147e-05, 6.05255779e-01, 6.47181232e-02, -1.99272433e-02, ],
        #               [-3.04741153e-04, 8.70794914e-05, 1.06955724e-05, 1.44813435e-03,
        #                1.15919271e-02, 2.09566793e-05, -1.97004768e-02, 5.52673548e-03,
        #                6.31241638e-04, 6.47181232e-02, 7.74670074e-01, -9.05845933e-03, ],
        #               [5.01076079e-06, -2.38874687e-05, -2.58476322e-07, -3.04261143e-04,
        #                -1.21550736e-04, 9.72255736e-06, 2.30538562e-04, -1.69964228e-03,
        #                -7.41514786e-06, -1.99272433e-02, -9.05845933e-03, 8.37034804e-04]])

        # Q = Q[6:, 6:]
        # print(Q.shape)
        # R = np.ones(12) * reg_weight
        R = np.ones(12) * reg_weight * 0.001

        quad_term = mass_matrix.T.dot(Q).dot(mass_matrix) + R
        linear_term = 1 * (g + desired_acc).T.dot(Q).dot(mass_matrix)
        return quad_term, linear_term

    def compute_contact_force(self,
                              foot_positions,
                              desired_acc,
                              contacts,
                              acc_weight=ACC_WEIGHT,
                              reg_weight=1e-4):
        mass_matrix = self.compute_mass_matrix(foot_positions)
        G, a = self.compute_objective_matrix(mass_matrix, desired_acc, acc_weight,
                                             reg_weight)
        C, b = self.compute_constraint_matrix(contacts)
        G += 1e-4 * np.eye(12)
        result = quadprog.solve_qp(G, a, C, b)
        return -result[0].reshape((4, 3))
