from unittest import TestCase, main
from torch import Tensor, randn, abs

from doublystochastic import DoublyStochastic, sinkhorn, rowColDiffNorm, randomNonNegTensor



class TestDoublyStochastic(TestCase):
    m = 10
    t = randomNonNegTensor(m)
    ds = DoublyStochastic(m)
    tol = 1e-6
    # print(f'{t}')
    def test_sinkhorn(self):
        tHat = sinkhorn(self.t, verbose=False, tol = self.tol)
        rn, cn = rowColDiffNorm(tHat)
        self.assertTrue(rn <= self.tol)
        self.assertTrue(cn <= self.tol)

if __name__ == '__main__':
    main()
