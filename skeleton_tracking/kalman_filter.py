
# Copyright 2024 National Research Council STIIMA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from numpy.linalg import multi_dot

# Constants
N_STATES = 9      # Number of states: [x,y,z,xp,yp,zp,xpp,ypp,zpp]
N_MEASURE = 3     # Number of measurements: [x,y,z]
IDENTITY = np.identity(N_STATES)
Q_DEFAULT = np.array([0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1]) / 90
R_DEFAULT = np.array([0.05, 0.05, 0.1]) / 90


class KalmanFilter:
    
    def __init__(self,
                 q_noise=Q_DEFAULT,
                 r_noise=R_DEFAULT,
                 dt=1.0 / 30.0):

        if len(q_noise) != N_STATES or len(r_noise) != N_MEASURE:
            raise ValueError('Invalid size of noise matrices')

        """ Class Builder """
        self.dt = dt
        self.t = None

        # Model matrix
        self.A = np.identity(N_STATES)
        self.A[0:3, 3:6] = np.identity(3) * self.dt
        self.A[3:6, 6:N_STATES] = np.identity(3) * self.dt
        self.A[0:3, 6:N_STATES] = np.identity(3) * (self.dt**2) * 0.5
        self.A[6:N_STATES, 6:N_STATES] = np.identity(3)

        self.C = np.zeros((N_MEASURE, N_STATES))
        self.C[0:3, 0:3] = np.identity(3)

        self.Q = np.identity(N_STATES) * q_noise

        self.R = np.identity(N_MEASURE) * r_noise

        self.P = None
        self.x_hat = None
        self.x_hat_new = None
        self.initialized = False

        self.y = None   # filtered x, y, z = C * x_hat_new
        self.skip_measure = 0

    def initialize(self, y_first_meas):
        """Initialize the Kalman filter."""
        self.t = 0.0
        P0 = np.ones((N_STATES, N_STATES)) * 1e-1
        self.P = P0

        self.x_hat = np.array([y_first_meas[0], y_first_meas[1],
                              y_first_meas[2], 0, 0, 0, 0, 0, 0])
        self.x_hat_new = self.x_hat
        self.y = np.array([y_first_meas[0], y_first_meas[1], y_first_meas[2]])

        self.initialized = True
        self.skip_measure = 0

        return self.initialized

    def get_y_after_initialize(self):
        return self.C.dot(self.x_hat)

    def open_loop_update(self):
        """Update without measurement (prediction only)."""
        self.x_hat_new = self.A.dot(self.x_hat)
        self.P = multi_dot([self.A, self.P, self.A.T]) + self.Q

        self.x_hat = self.x_hat_new
        self.t += self.dt
        return self.C.dot(self.x_hat_new)

    def update(self, y_measure, id_keypoint):
        """
        Update Kalman estimation.

        @param: y_measure: measurements [xm, ym, zm]
        """
        self.x_hat_new = self.A.dot(self.x_hat)
        self.y_observed_priori = self.C.dot(self.x_hat_new)

        if abs(self.y_observed_priori[2] - y_measure[2]
               ) > 0.5 and self.skip_measure < 3 and self.initialized:
            self.skip_measure += 1
            return self.open_loop_update()
        elif self.skip_measure >= 3:
            self.initialize(y_measure)

            print('-------------------------Reinitializing--------------------')
            print('Keypoint: ', id_keypoint)
            print('Measurement: {}'.format(y_measure))

            return self.get_y_after_initialize()
        else:
            self.P = multi_dot([self.A, self.P, self.A.T]) + self.Q

            try:
                K = multi_dot([
                    self.P, self.C.T,
                    np.linalg.inv(multi_dot([self.C, self.P, self.C.T]) + self.R)
                ])
            except np.linalg.LinAlgError as e:
                raise(e)

            self.x_hat_new += K.dot(y_measure - self.y_observed_priori)
            self.P = (IDENTITY - K.dot(self.C)).dot(self.P)

            self.y_observed_posteriori = self.C.dot(self.x_hat_new)
            self.y = self.y_observed_posteriori
            self.x_hat = self.x_hat_new
            self.t += self.dt

            self.skip_measure = 0
            return self.y_observed_posteriori

    def get_cartesian_velocity(self):
        """Return estimated velocity [vx, vy, vz]."""
        return self.x_hat_new[3:6]

    def get_cartesian_acceleration(self):
        """Return estimated acceleration [ax, ay, az]."""
        return self.x_hat_new[6:]

    def get_covariance(self):
        return np.diagonal(self.P)

    def get_pos_dev_st(self):
        """Return standard deviation of position estimate [dev_x, dev_y, dev_z]."""
        return np.sqrt(np.diagonal(self.P))[0:3]

    def get_vel_dev_st(self):
        """Return standard deviation of velocity estimate [dev_vx, dev_vy, dev_vz]."""
        return np.sqrt(np.diagonal(self.P))[3:6]

    def get_acc_dev_st(self):
        """Return standard deviation of acceleration estimate [dev_ax, dev_ay, dev_az]."""
        return np.sqrt(np.diagonal(self.P))[6:]

    def reset(self):
        """Reset the Kalman filter."""
        self.x_hat = np.zeros((N_STATES,))
        self.t = 0
        self.initialized = False

    def get_state_estimate(self):
        return self.x_hat

    def get_output_estimate(self):
        return self.y
