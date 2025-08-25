import unittest
import sympy as sp
from src.algebra.tanaka_symbols.symplectic_symbol import SympSymb
from src.distributions import Distr_of_constant_symbol

class abn_ind_der_test(unittest.TestCase):
    def setUp(self):
        self.T7=SympSymb(7)
        self.C7=self.T7.cochain_complex
        self.D7=Distr_of_constant_symbol(self.T7,self.C7.elt({}))
        # self.startTime = time.time()
        
    # def tearDown(self):
    #     t = time.time() - self.startTime
    #     print('%s: %.3f' % (self.id(), t))
    
    def test_indexed_obj(self):
        K=sp.IndexedBase('K')
        self.assertEqual(self.D7.abn_ind_der(K[0],3),K[0,3])
        self.assertEqual(self.D7.abn_ind_der(K[0,4,3,2],4),K[0,4,3,2,4])
        self.assertEqual(self.D7.abn_ind_der(K[0,1]*K[0,2],5),K[0,1,5]*K[0,2]+K[0,1]*K[0,2,5])
        self.assertEqual(self.D7.abn_ind_der(K[0]*K[1]**2,3),K[0,3]*K[1]**2+2*K[0]*K[1,3]*K[1])
        self.assertEqual(self.D7.abn_ind_der(self.D7.abn_ind_der(K[5,5,5,5],4),3),K[5,5,5,5,4,3])
        
        
    def test_coord(self):
        h,e,y=sp.symbols('h,e,y')
        self.assertEqual(self.D7.abn_ind_der(y,3),0)
        self.assertEqual(self.D7.abn_ind_der(h,3),0)
        self.assertEqual(self.D7.abn_ind_der(e,3),0)
        self.assertEqual(self.D7.abn_ind_der(y*h,5),0)
        self.assertEqual(self.D7.abn_ind_der(e**2,4),0)
        
    def test_Leib_and_sums(self):
        K=sp.IndexedBase('K')
        h,e,y=sp.symbols('h,e,y')
        A=h*K[0]*K[1]**2
        dA=h*K[0,3]*K[1]**2+2*h*K[0]*K[1]*K[1,3]
        self.assertEqual(self.D7.abn_ind_der(A,3),dA)
        self.assertEqual(self.D7.abn_ind_der(h+y+e,4),0)
        self.assertEqual(self.D7.abn_ind_der(h+K[0,1,2],3),K[0,1,2,3])
        self.assertEqual(self.D7.abn_ind_der(h*y+K[1]*e**2,4),e**2*K[1,4])
        self.assertEqual(self.D7.abn_ind_der(K[1]+4*y**2+e,5),K[1,5])