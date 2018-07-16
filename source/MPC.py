import numpy as np
from scipy.linalg import block_diag

class MPCTracker:
    def tracker(self, A, B, C, Q, R, S, ref, x0, u0, nstep):
        # Linear model
        #  xk+1 = A*xk + B* u
        #  yk = C*xk (No feed forward)

        # Augment the delta control input delta_u
        A_tilda, B_tilda = self.concat(A, B, C, nstep)

    def concat(self, A, B, C, dimU):
        # augment system
        # A_tilda = [[A B]
        #            [0 I]]
        # B_tilda = [[B]
        #            [I]]
        fill = np.concatenate((np.zeros((dimU, A.shape[1])), np.eye(dimU,B.shape[1])), axis=1)
        A_tilda = np.concatenate((np.concatenate((A, B), axis=1), fill), axis=0)
        B_tilda = np.concatenate((B, np.ones([dimU, B.shape[1]])), axis=0)
        C_tilda = np.concatenate((C, np.zeros([C.shape[0], dimU])), axis=1)

        # J = 0.5*err'*S*err + 0.5*sum(err)

        return A_tilda, B_tilda, C_tilda
    def getDoubleBar(self,Q, R, S, A_tilda, B_tilda, C_tilda, nstep=2):
        Qdbar = Q
        QC = np.dot(Q, C_tilda)
        Tdbar = QC
        Rdbar = R
        Adbar = A_tilda
        Bdbar = B_tilda
        A0dbar = A_tilda
        for i in range(nstep-2):
            Qdbar = block_diag(Qdbar, Q)
            Tdbar = block_diag(Tdbar, QC)
            Rdbar = block_diag(Rdbar, R)
            Adbar = block_diag(Adbar, A_tilda)
            Bdbar = block_diag(Bdbar, B_tilda)
            A0dbar = np.concatenate((A0dbar,np.zeros(A_tilda.shape)),axis = 0)

        Qdbar = block_diag(Qdbar, S)
        Tdbar = block_diag(Tdbar, np.dot(S, C_tilda))
        Rdbar = block_diag(Rdbar, R)
        Bdbar = block_diag(Bdbar, B_tilda)
        A0dbar = np.concatenate((A0dbar, np.zeros([A_tilda.shape[0], A_tilda.shape[1]])), axis=0)
        Adbar = np.concatenate((Adbar, np.zeros([Adbar.shape[0], A_tilda.shape[0]])), axis=1)
        Adbar = np.concatenate((np.zeros([A_tilda.shape[0], Adbar.shape[1]]), Adbar), axis=0)
        return Qdbar, Rdbar, Tdbar, Adbar, Bdbar, A0dbar
