import unittest
import sympy as sp
from symp_dist.algebra.tanaka_symbols.symplectic_symbol import SympSymb

class TestCb(unittest.TestCase):
    # To do: test this more thoroughly!
    def setUp(self):
        self.T=SympSymb(5)
        self.C=self.T.cochain_complex
    
    def test_d2_zero(self):
        for d in [1,2]:
            for w in self.C.basis(d):
                A=self.C.cb_mat(d+1,w)*self.C.cb_mat(d,w)
                self.assertEqual(A,sp.zeros(*sp.shape(A)))
                
    def test_cb_on_base(self):
        c0=self.C.elt({})
        c1=self.C.elt_from_cd({('Y',):1})
        c2=self.C.elt_from_cd({('e_2','E'):1})
        c3=self.C.elt_from_cd({('X','X'):1})
        c4=self.C.elt_from_cd({('e_3','X','Y'):1})
        c5=self.C.elt_from_cd({('e_3','e_2','X'):1})
        c6=self.C.elt_from_cd({('N','N'):1})
        cs=[c0,c1,c2,c3,c4,c5,c6]
        
        d0=self.C.elt({})
        d1=self.C.elt_from_cd({('X','H'):1,('e_4','e_3'):-3,('e_3','e_2'):-4,('e_2','e_1'):-3})
        d2=self.C.elt_from_cd({('e_2','e_1','e_1'):1,('e_2','e_3','e_3'):1,('e_2','e_4','e_4'):1,('e_2','N','N'):2,('X','e_1','E'):-1})
        d3=self.C.elt_from_cd({('X','e_1','e_2'):1,('X','e_2','e_3'):1,('X','e_3','e_4'):1})
        d4=self.C.elt_from_cd({('e_3','X','e_4','e_3'):-3,('e_3','X','e_2','e_1'):-3})
        d5=self.C.elt_from_cd({('e_3','e_2','e_1','e_2'):-1,('X','e_1','e_3','X'):1})
        d6=self.C.elt_from_cd({('e_1','e_4','N'):1,('e_2','e_3','N'):-1})
        ds=[d0,d1,d2,d3,d4,d5,d6]
        
        for i in range(len(cs)):
            self.assertEqual(cs[i].cb(),ds[i])
            
    def test_cb_on_lin_combs(self):
        c0=self.C.elt({})
        c1=self.C.elt_from_cd({('Y',):1})
        c2=self.C.elt_from_cd({('e_2','E'):1})
        c3=self.C.elt_from_cd({('X','X'):1})
        c4=self.C.elt_from_cd({('e_3','X','Y'):1})
        c5=self.C.elt_from_cd({('e_3','e_2','X'):1})
        c6=self.C.elt_from_cd({('N','N'):1})
        cs=[c0,c1,c2,c3,c4,c5,c6]
        
        d0=self.C.elt({})
        d1=self.C.elt_from_cd({('X','H'):1,('e_4','e_3'):-3,('e_3','e_2'):-4,('e_2','e_1'):-3})
        d2=self.C.elt_from_cd({('e_2','e_1','e_1'):1,('e_2','e_3','e_3'):1,('e_2','e_4','e_4'):1,('e_2','N','N'):2,('X','e_1','E'):-1})
        d3=self.C.elt_from_cd({('X','e_1','e_2'):1,('X','e_2','e_3'):1,('X','e_3','e_4'):1})
        d4=self.C.elt_from_cd({('e_3','X','e_4','e_3'):-3,('e_3','X','e_2','e_1'):-3})
        d5=self.C.elt_from_cd({('e_3','e_2','e_1','e_2'):-1,('X','e_1','e_3','X'):1})
        d6=self.C.elt_from_cd({('e_1','e_4','N'):1,('e_2','e_3','N'):-1})
        ds=[d0,d1,d2,d3,d4,d5,d6]

        # Test some linear combinations, too
        LCs=[[3,-2,1,3,4,5,4],[0,0,0,-1,-1,2,-6],[4,5,sp.Rational(1,4),sp.Rational(-2,7),100,-8,0]]
        for LC in LCs:
            c=sum([LC[i]*cs[i] for i in range(len(LC))],start=self.C.elt({}))
            d=sum([LC[i]*ds[i] for i in range(len(LC))],start=self.C.elt({}))
            self.assertEqual(c.cb(),d)