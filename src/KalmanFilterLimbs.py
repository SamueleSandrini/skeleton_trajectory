import numpy as np
from numpy.linalg import multi_dot
from math import sin, cos, asin, acos, atan2

#Constant
N_STATES=14      # Number of states: [q1,q2,q3,q4,q1p,q2p,q3p,q4p,q1pp,q2pp,q3pp,q4pp]
N_MEASURE=6     # Number of measurements: [xg,yg,zg, xp,yp,zp]
R_0=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
Z_GAIN=1.2

I = np.identity(N_STATES)

class KalmanFilterLimbs():
    def __init__(self):
        """ Class Builder """
        self.dt = 1.0/30.0  # Sampling Time
        self.t = None

        #Model matrix
        self.A=np.identity(N_STATES)
        self.A[0:4,4:8]=np.identity(4)*self.dt
        self.A[4:8,8:12]=np.identity(4)*self.dt
        self.A[0:4,8:12]=np.identity(4)*(self.dt**2)*0.5

        self.C = np.zeros((N_MEASURE, N_STATES))


        q =  np.array([10, 10, 10, 10, 50, 50, 50,50, 100, 100, 100, 100, 1, 1])
        #q =  np.array([1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6,1e6, 1e6, 1e6, 1e6, 1e6 ])
        self.Q = np.identity(N_STATES)*q            #[q_0 0 0 ..., 0 q_1 0 0, 0 0 q_2 ... ]

        self.R=np.identity(N_MEASURE)*R_0 # x y z x y z
        #self.R=np.identity(N_MEASURE)*[1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6] # x y z x y z

        self.P = None

        self.x_hat = None
        self.x_hat_new = None
        self.initialized = False

        self.y = None
        self.skip_measure = 0
        self.err=0

    def initialize(self,yFirstMeas,zVect):
        """
        Method for initialize the Kalman
        @param yFirstMeas: Firs measurement [xg,yg,zg, xp,yp,zp]
        @param zVect: zVector for reproject z Variance of depth (camera frame) to x,y,z (local frame)
        """
        print("------------------------INITIALIZE------------------------------------")

        #Initial settings
        self.t=0.0

        P0 = np.ones((N_STATES,N_STATES))*1e-1
        self.P = P0

        self.R=np.identity(N_MEASURE)*(R_0 + np.abs(np.tile(zVect, 2))*Z_GAIN)

        #Calculation about limbs length
        self.l1=np.linalg.norm(yFirstMeas[0:3])
        self.l2=np.linalg.norm(yFirstMeas[0:3] - yFirstMeas[3:])    #3: dal terzo all'ultimo
        # self.l1=0.28
        # self.l2=0.25
        print("L1=",self.l1)
        print("L2=",self.l2)

        #First state
        self.x_hat = np.concatenate((self.inverseKinematic(yFirstMeas),[0,0,0,0,0,0,0,0,self.l1,self.l2]))    # yFirstMeas=[xg, yg, zg, xp, yp, zp]
        self.x_hat_new=self.x_hat
        self.C = self.hJacobian(self.x_hat)

        #Set initialized flag

        self.skip_measure=0
        #self.initialized = True
        #return self.initialized

        if self.l1<0.7 and self.l1>0.1 and self.l2<0.7 and self.l2>0.1:
            self.initialized = True
            return self.initialized
        else:
            self.initialized = False
            return self.initialized

    def getYAfterInitialize(self):
        return self.forwardKinematic(self.x_hat)

    def updateOpenLoop(self):
        """  Method for update without Kalman but only based on model  """
        # State a-priori estimation
        self.x_hat_new = self.A.dot(self.x_hat)

        # La matrice P è comunque da aggiornare
        self.P = multi_dot( [self.A , self.P , self.A.T ]) + self.Q

        # Update
        self.x_hat = self.x_hat_new
        self.t +=self.dt

        return self.forwardKinematic(self.x_hat_new)

    def update(self,yMeasure,zVect):
        """
        Method for update Kalman estimation
         @param: y: measurements [xg, yg, zg, xp, yp, zp]' numpy array
         @param zVect: zVector for reproject z Variance of depth (camera frame) to x,y,z (local frame)
        """
        self.R=np.identity(N_MEASURE)*(R_0 + np.abs(np.tile(zVect, 2))*Z_GAIN)

        #print("L1=",self.l1)
        #print("L2=",self.l2)
        # State a-priori estimation
        self.x_hat_new = self.A.dot(self.x_hat)
        self.l1=self.x_hat_new[-2]
        self.l2=self.x_hat_new[-1]
        self.y_observed_priori = self.forwardKinematic(self.x_hat_new)


        self.P = multi_dot( [self.A , self.P , self.A.T ]) + self.Q

        self.err = yMeasure - self.y_observed_priori

        self.C = self.hJacobian(self.x_hat_new)

        try:
            K = multi_dot([self.P, self.C.T, np.linalg.inv( multi_dot([self.C , self.P, self.C.T]) +self.R) ])
        except np.linalg.LinAlgError as e:
            raise

        # A-posteriori correction
        self.x_hat_new += K.dot( yMeasure - self.y_observed_priori )
        self.l1=self.x_hat_new[-2]
        self.l2=self.x_hat_new[-1]
        self.P = (I - K.dot( self.C )).dot(self.P)

        self.y_observed_posteriori = self.forwardKinematic(self.x_hat_new)

        # Update
        self.x_hat = self.x_hat_new
        self.t +=self.dt

        self.skip_measure=0
        return self.y_observed_posteriori

    def getJointsPosition(self):
        #print(self.x_hat_new)
        return self.x_hat_new[0:4]
    def getJointsVelocity(self):
        return self.x_hat_new[4:8]
    def getJointsAcceleration(self):
        return self.x_hat_new[8:12]
    def getLimbLength(self):
        return self.x_hat_new[12:]

    def reset(self):
        """
        Method for reset the kalman filter
        """
        self.x_hat = np.zeros((N_STATES,))
        self.t = 0
        self.initialized=False

    def hJacobian(self,q):
        """
        Method for evaluate the Jacobian Matrix
        @param q: Vector containing all joints variable (also state vector is goog: first 4 elements)
        """
        J = np.zeros([6,14],dtype=float)  #Dim: 6x12
        J[0,0] = self.l1*cos(q[0])*cos(q[1])
        J[0,1] = -self.l1*sin(q[0])*sin(q[1])
        J[1,0] = self.l1*cos(q[1])*sin(q[0])
        J[1,1] = self.l1*cos(q[0])*sin(q[1])
        J[2,1] = -self.l1*cos(q[1])
        J[3,0] = self.l1*cos(q[0])*cos(q[1]) - self.l2*(sin(q[3])*(cos(q[2])*sin(q[0]) + cos(q[0])*sin(q[1])*sin(q[2])) - cos(q[0])*cos(q[1])*cos(q[3]))
        J[3,1] = -sin(q[0])*(self.l1*sin(q[1]) + self.l2*cos(q[3])*sin(q[1]) + self.l2*cos(q[1])*sin(q[2])*sin(q[3]))
        J[3,2] = -self.l2*sin(q[3])*(cos(q[0])*sin(q[2]) + cos(q[2])*sin(q[0])*sin(q[1]))
        J[3,3] = self.l2*(cos(q[3])*(cos(q[0])*cos(q[2]) - sin(q[0])*sin(q[1])*sin(q[2])) - cos(q[1])*sin(q[0])*sin(q[3]))
        J[4,0] = self.l2*(sin(q[3])*(cos(q[0])*cos(q[2]) - sin(q[0])*sin(q[1])*sin(q[2])) + cos(q[1])*cos(q[3])*sin(q[0])) + self.l1*cos(q[1])*sin(q[0])
        J[4,1] = cos(q[0])*(self.l1*sin(q[1]) + self.l2*cos(q[3])*sin(q[1]) + self.l2*cos(q[1])*sin(q[2])*sin(q[3]))
        J[4,2] = -self.l2*sin(q[3])*(sin(q[0])*sin(q[2]) - cos(q[0])*cos(q[2])*sin(q[1]))
        J[4,3] = self.l2*(cos(q[3])*(cos(q[2])*sin(q[0]) + cos(q[0])*sin(q[1])*sin(q[2])) + cos(q[0])*cos(q[1])*sin(q[3]))
        J[5,1] = -self.l2*(cos(q[1])*cos(q[3]) - sin(q[1])*sin(q[2])*sin(q[3])) - self.l1*cos(q[1])
        J[5,2] = -self.l2*cos(q[1])*cos(q[2])*sin(q[3])
        J[5,3] = self.l2*(sin(q[1])*sin(q[3]) - cos(q[1])*cos(q[3])*sin(q[2]))
        #Aggiunta ultime due colonne per l
        J[0,-2] = cos(q[1])*sin(q[0])
        J[1,-2] = -cos(q[0])*cos(q[1])
        J[2,-2] = -sin(q[1])
        J[3,-2] = cos(q[1])*sin(q[0])
        J[3,-1] = sin(q[3])*(cos(q[0])*cos(q[2]) - sin(q[0])*sin(q[1])*sin(q[2])) + cos(q[1])*cos(q[3])*sin(q[0])
        J[4,-2] = -cos(q[0])*cos(q[1])
        J[4,-1] = sin(q[3])*(cos(q[2])*sin(q[0]) + cos(q[0])*sin(q[1])*sin(q[2])) - cos(q[0])*cos(q[1])*cos(q[3])
        J[5,-2] = -sin(q[1])
        J[5,-1] = - cos(q[3])*sin(q[1]) - cos(q[1])*sin(q[2])*sin(q[3])

        return J

    def forwardKinematic(self,q):
        """
        Method for evaluate forward kinematic
        @param q: Vector containing all joints variable (also state vector is goog: first 4 elements)
        """
        Fk = np.empty([6,],dtype=float)
        Fk[0] = self.l1*cos(q[1])*sin(q[0])
        Fk[1] = -self.l1*cos(q[0])*cos(q[1])
        Fk[2] = -self.l1*sin(q[1])
        Fk[3] = self.l2*(sin(q[3])*(cos(q[0])*cos(q[2]) - sin(q[0])*sin(q[1])*sin(q[2])) + cos(q[1])*cos(q[3])*sin(q[0])) + self.l1*cos(q[1])*sin(q[0])
        Fk[4] = self.l2*(sin(q[3])*(cos(q[2])*sin(q[0]) + cos(q[0])*sin(q[1])*sin(q[2])) - cos(q[0])*cos(q[1])*cos(q[3])) - self.l1*cos(q[0])*cos(q[1])
        Fk[5] = - self.l2*(cos(q[3])*sin(q[1]) + cos(q[1])*sin(q[2])*sin(q[3])) - self.l1*sin(q[1])
        return Fk

    def inverseKinematic(self,X):
        """
        Method for evaluate inverse Kinematic
        @param X: vector containing [xg,yg,zg,xp,yp,zp] in local frame
        """
        q = np.empty([4,],dtype=float)

        #First two joint angles
        arg1=-X[2]/self.l1
        if arg1>1:
            arg1=1

        #print(-X[4]/self.l2)
        q[0] = atan2(X[0],-X[1])    # atan2(xg,-yg)
        q[1] = asin(arg1)

        PforG=self.PrespectG(q[0],q[1],X[3:])

        #print("Pfor0 ", X[3:])
        #print("PforG ", PforG)
        XforInv=np.concatenate([X[0:3], PforG])
        arg3=-XforInv[4]/self.l2
        if arg3>1:
            arg3=1
        q[2] = atan2(XforInv[5],-XforInv[3])
        #print(XforInv[4])
        #print(arg3)
        q[3] = 2*np.pi-acos(arg3)

        return q

    def PrespectG(self,q1,q2,P):
        """
        Method for evaluate P respect to G, usefull for inverse kinematic
        @param q1: q1 joint angle
        @param q2: q2 joint angle
        @param P: P coordinate [xp,yp,zp]
        """
        M02=np.zeros([4,4])
        M02[0,0]=cos(q1)
        M02[0,1]=-cos(q2)*sin(q1)
        M02[0,2]= sin(q1)*sin(q2)
        M02[0,3]= self.l1*cos(q2)*sin(q1)
        M02[1,0]=sin(q1)
        M02[1,1]=cos(q1)*cos(q2)
        M02[1,2]=-cos(q1)*sin(q2)
        M02[1,3]=-self.l1*cos(q1)*cos(q2)
        M02[2,1]=sin(q2)
        M02[2,2]=cos(q2)
        M02[2,3]=-self.l1*sin(q2)
        M02[-1,-1]=1

        #Aggiungo 1 : coordinata omogenea
        P=np.concatenate([P,[1]])

        PrespectGext=np.dot(np.linalg.inv(M02),P)
        #print(PrespectGext[0:3])
        try:
            return PrespectGext[0:3]
        except np.linalg.LinAlgError as e:
            raise

    def getCartesianVelocity(self):
        return np.dot(self.C[:,0:4],self.x_hat_new[4:8])

    def getCartesianCovariance(self):
        """
        Getter method for giving keypoint standard deviation of keypoint position
        @ return: [dev_st x, dev_st y, dev_st z]
        """
        cartCov=np.diagonal(multi_dot( [self.C[:,0:4] , self.P[0:4,0:4] , self.C[:,0:4].T ]))
        return cartCov
