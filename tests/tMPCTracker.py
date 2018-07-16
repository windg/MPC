import unittest
import numpy as np
from source.MPC import MPCTracker
from numpy import testing as nt
class TestTracker(unittest.TestCase):
    def setUp(self):
        self.mpc = MPCTracker()
    def test_concat(self):
        dimU = 4
        dimState = 3
        dimOutput = 3
        A = np.random.rand(dimState, dimState)
        B = np.random.rand(dimState, dimU)
        C = np.random.rand(dimOutput, dimState)

        A_tilda, B_tilda, C_tilda = self.mpc.concat(A, B, C, dimU)
        expAdim = (dimState+dimU,dimState+dimU)
        expBdim = (dimState+dimU,dimU)
        expCdim = (dimOutput, dimState+dimU)
        nt.assert_array_equal(A_tilda.shape, expAdim)
        nt.assert_array_equal(B_tilda.shape, expBdim)
        nt.assert_array_equal(C_tilda.shape, expCdim)
    def test_getDoubleBar(self):
        dimU = 1
        dimState = 2
        dimOutput = 1
        A = np.eye(dimState, dimState)
        B = np.ones((dimState, dimU))
        C = np.ones((dimOutput, dimState))
        S = 2*np.eye(dimOutput,dimOutput)
        Q = 3*np.eye(dimOutput,dimOutput)
        R = np.eye(dimU,dimU)
        Qdbar, Rdbar, Tdbar, Adbar, Bdbar, A0dbar = \
            self.mpc.getDoubleBar(Q=Q, R=R, S=S, A_tilda=A, B_tilda=B, C_tilda=C, nstep=4)
        print('Qd:\n', Qdbar)
        print('Rd:\n', Rdbar)
        print('Td:\n', Tdbar)
        print('Ad:\n', Adbar)
        print('Bd:\n', Bdbar)
        print('A0d:\n', A0dbar)



if __name__ == '__main__':
    unittest.main()