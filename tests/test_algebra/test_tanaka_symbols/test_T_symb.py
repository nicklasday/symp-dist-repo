import unittest
import sympy as sp
from symp_dist.algebra.tanaka_symbols.symplectic_symbol import SympSymb
from symp_dist.utils.exceptions import heis_dim_exception
from symp_dist.algebra.tanaka_symbols.tanaka_symbol_basis_elt import TSymbBasisElt

class test_T_symb(unittest.TestCase):
    def setUp(self):
        self.T3=SympSymb(3)
        self.T5=SympSymb(5)
        self.T7=SympSymb(7)
        self.T9=SympSymb(9)
#         self.startTime = time.time()
        
#     def tearDown(self):
#         t = time.time() - self.startTime
#         print('%s: %.3f' % (self.id(), t))
        
    def test_init(self):
        self.assertRaises(heis_dim_exception,SympSymb,4)
        self.assertRaises(heis_dim_exception,SympSymb,0)
        
        self.assertEqual(self.T3.heis_dim,3)
        self.assertEqual(self.T3.basis_strs,['Y','H','E','X','e_1','e_2','N'])
        self.assertEqual(len(self.T3.basis),7)
        for A in self.T3.basis:
            self.assertEqual(type(A),TSymbBasisElt)
            

        X=self.T7.basis[3]
        self.assertEqual(X.vec,sp.Matrix([0]*3+[1]+[0]*(len(self.T7.basis)-4)))
        self.assertEqual(X.wght,-1)
        self.assertEqual(X.parent,self.T7)
        
        # This takes about 8 seconds. Should that be concerning?
        self.assertTrue(self.T3.jacobi_test())
        self.assertTrue(self.T5.jacobi_test())
        self.assertTrue(self.T7.jacobi_test())
        self.assertTrue(self.T9.jacobi_test())
        
    def test_elt_init(self):
        elt0=self.T3.elt([0]*len(self.T3.basis))
        elt1=self.T7.elt([0]*len(self.T7.basis))
        elt2=self.T3.elt([1,-1]+[0]*(len(self.T3.basis)-2))
        elt3=self.T3.elt()
        e2=self.T7.elt([0,0,0,0,0,1]+[0]*(len(self.T7.basis)-6))
        
        self.assertEqual(elt0,elt3)
        self.assertNotEqual(elt0,elt1)
        self.assertEqual(str(elt0),'0')
        self.assertEqual(e2,self.T7.basis[5])
        
        self.assertEqual(elt2,self.T3.basis[0]-self.T3.basis[1])
        self.assertEqual(str(elt2),'Y - H')
        
    def test_iprod(self):
        Y,H,E,X,e1,e2,e3,e4,e5,e6,N=self.T7.basis
        oh=self.T7.elt()
        v1=X+4*Y
        v2=Y+2*e2
        v3=-3*N-e2
        v4=X+N
        
        self.assertEqual(oh.iprod(oh),0)
        self.assertEqual(oh.iprod(X),0)
        self.assertEqual(X.iprod(oh),0)
        self.assertEqual(oh.iprod(v1),0)
        self.assertEqual(v1.iprod(oh),0)
        
        self.assertEqual(Y.iprod(Y),1)
        self.assertEqual(H.iprod(H),2)
        self.assertEqual(E.iprod(E),2)
        self.assertEqual(X.iprod(X),1)
        self.assertEqual(e1.iprod(e1),1/sp.factorial(5))
        self.assertEqual(e2.iprod(e2),5/sp.factorial(5))
        self.assertEqual(e3.iprod(e3),40/sp.factorial(5))
        self.assertEqual(e6.iprod(e6),sp.factorial(5))
        
        self.assertEqual(v1.iprod(v2),4)
        self.assertEqual(v1.iprod(v3),0)
        self.assertEqual(v1.iprod(v4),1)
        self.assertEqual(v2.iprod(v3),-10/sp.factorial(5))
        self.assertEqual(v2.iprod(v4),0)
        self.assertEqual(v3.iprod(v4),-3)
