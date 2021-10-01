import numpy as np
from numpy.linalg import multi_dot

#Constant
N_STATES=9      # Number of states: [x,y,z,xp,yp,zp,xpp,ypp,zpp]
N_MEASURE=3     # Number of measurements: [x,y,z]
I = np.identity(N_STATES)

class KalmanFilter():
    def __init__(self):
        """ Class Builder """
        self.dt = 1.0/30.0
        self.t = None
        #Model matrix
        self.A=np.identity(N_STATES)
        self.A[0:3,3:6]=np.identity(3)*self.dt
        self.A[3:6,6:np.size(self.A,1)]=np.identity(3)*self.dt
        self.A[0:3,6:np.size(self.A,1)]=np.identity(3)*(self.dt**2)*0.5
        self.A[6:np.size(self.A,0),6:np.size(self.A,1)]=np.identity(3)

        self.C = np.zeros((N_MEASURE, N_STATES))
        self.C[0:3,0:3]=np.identity(3)
        #q =  np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        q =  np.array([0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1])/90
        self.Q = np.identity(N_STATES)*q            #[q_0 0 0 ..., 0 q_1 0 0, 0 0 q_2 ... ]
        #print(self.Q)

        self.R=np.identity(N_MEASURE)*[0.05, 0.05, 0.1]/90
        #self.R = np.array([1, 1, 10])
        #print(self.R)
        self.P = None

        self.x_hat = None
        self.x_hat_new = None
        self.initialized = False

        self.y = None   #I would like to add also y : x,y,z filtrati C*x_hat_new
        self.skip_measure = 0

    def initialize(self,yFirstMeas):
        """
        Method for initialize the Kalman
        """

        #Initial settings
        self.t=0.0
        P0 = np.ones((N_STATES,N_STATES))*1e-1
        self.P = P0

        #First state
        self.x_hat = np.array([yFirstMeas[0],yFirstMeas[1],yFirstMeas[2], 0,0,0, 0,0,0])    #x,y,z like measured, vel and acc = 0
        self.x_hat_new=self.x_hat

        #Set initialized flag
        self.initialized = True
        self.skip_measure=0

        return self.initialized

    def getYAfterInitialize(self):
        return self.C.dot(self.x_hat)

    def updateOpenLoop(self):
        """
        Method for update without Kalman but only based on model
        """

        # State a-priori estimation
        self.x_hat_new = self.A.dot(self.x_hat)
        # La matrice P è comunque da aggiornare: da vedere come!
        self.P = multi_dot( [self.A , self.P , self.A.T ]) + self.Q

        # Kalman for Intermittent Observation : https://arxiv.org/pdf/0903.2890.pdf, https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1333199

        # Update
        self.x_hat = self.x_hat_new
        self.t +=self.dt
        return self.C.dot(self.x_hat_new)
    def update(self,yMeasure,idKeypoint):
        """
        Method for update Kalman estimation
         @param: y: measurements [xm,ym,zm]' numpy array
        """
        # State a-priori estimation
        self.x_hat_new = self.A.dot(self.x_hat)
        self.y_observed_priori = self.C.dot(self.x_hat_new)
        if abs(self.y_observed_priori[2]-yMeasure[2])>0.5 and self.skip_measure<3 and self.initialized:
            #print("Skip_measure = ", self.skip_measure)
            #print("Keipoint: ", idKeypoint)
            #print("Misura: {}".format(yMeasure))
            self.skip_measure+=1
            return self.updateOpenLoop()
        elif self.skip_measure>=3:      #and abs(self.y_observed_priori[2]-yMeasure[2] )>0.5
            self.initialize(yMeasure)

            print("-------------------------Reinizializzo--------------------")
            print("Keipoint: ",idKeypoint)
            print("Misura: {}".format(yMeasure))

            return self.getYAfterInitialize()
        else:
            self.P = multi_dot( [self.A , self.P , self.A.T ]) + self.Q

            # Rt = self.R se  yMeasure - self.C.dot(self.x_hat_new)  piccolo, oppure self.Rlost
            try:
                K = multi_dot([self.P, self.C.T, np.linalg.inv( multi_dot([self.C , self.P, self.C.T]) +self.R) ])
            except np.linalg.LinAlgError as e:
                raise

            # A-posteriori correction
            self.x_hat_new += K.dot( yMeasure - self.y_observed_priori )
            self.P = (I - K.dot( self.C )).dot(self.P)

            self.y_observed_posteriori = self.C.dot(self.x_hat_new)     #Adding by me
            # Update
            self.x_hat = self.x_hat_new
            self.t +=self.dt

            self.skip_measure=0
            return self.y_observed_posteriori

    def getCartesianVelocity(self):
        """
        Getter method for giving the keypoint cartesian velocity (vx, vy, vz)
        @ return: [vx,vy,vz]
        """
        return self.x_hat_new[3:6]
    def getCartesianAcceleration(self):
        """
        Getter method for giving keypoint cartesian acceleration (ax, ay, az)
        @ return: [ax,ay,az]
        """
        return self.x_hat_new[6:]

    def getCovariance(self):
        return np.diagonal(self.P)

    def getPosDevSt(self):
        """
        Getter method for giving keypoint standard deviation of keypoint position
        @ return: [dev_st x, dev_st y, dev_st z]
        """
        return np.sqrt(np.diagonal(self.P))[0:3]
    def getVelDevSt(self):
        """
        Getter method for giving keypoint standard deviation of keypoint velocity
        @ return: [dev_st vx, dev_st vy, dev_st vz]
        """
        return np.sqrt(np.diagonal(self.P))[3:6]
    def getAccDevSt(self):
        """
        Getter method for giving keypoint standard deviation of keypoint acceleration
        @ return: [dev_st ax, dev_st ay, dev_st az]
        """
        return np.sqrt(np.diagonal(self.P))[6:]

    def reset(self):
        """
        Method for reset the kalman filter
        """
        this.x_hat = np.zeros((N_STATES,))
        this.t = 0
        self.initialized=False
