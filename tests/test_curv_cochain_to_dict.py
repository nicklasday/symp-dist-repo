import unittest
import sympy as sp
from src.algebra.tanaka_symbols.symplectic_symbol import SympSymb
from src.distributions import Free_distr
from src.cartan_geometries import Geom_Prolongation

class test_curv_cochain_to_dict(unittest.TestCase):
    def setUp(self):
        self.K=sp.IndexedBase('K')
        self.T=SympSymb(5)
        self.C=self.T.cochain_complex
        self.D=Free_distr(self.T)[0]
        self.P=Geom_Prolongation(self.D,{})

    def test_curv_cochain_to_dict(self):
        c0=self.C.elt({})
        c1=self.C.elt_from_cd({('X','e_1','e_2'):1})
        c2=self.C.elt_from_cd({('X','e_1','e_3'):1})
        c3=self.C.elt_from_cd({('e_2','e_3','N'):1})
        c4=3*c1-5*c2
        c5=-c1+c2-4*c3

        e2=sp.zeros(len(self.T.basis),1)
        e3=sp.zeros(len(self.T.basis),1)
        N=sp.zeros(len(self.T.basis),1)
        e2[5]=1
        e3[6]=1
        N[-1]=1

        r0={}
        r1={}
        r2={}
        r3={}
        r4={}
        r5={}

        for r in [r0,r1,r2,r3,r4,r5]:
            for i in range(3,len(self.T.basis)):
                for j in range(i+1,len(self.T.basis)):
                    r[(i,j)]=sp.zeros(len(self.T.basis),1)

        r1[(3,4)]=e2
        r2[(3,4)]=e3
        r3[(5,6)]=N
        r4[(3,4)]=3*e2-5*e3
        r5[(3,4)]=-e2+e3
        r5[(5,6)]=-4*N
        
        self.assertEqual(self.C.curv_cochain_to_dict(c0),r0)
        self.assertEqual(self.C.curv_cochain_to_dict(c1),r1)
        self.assertEqual(self.C.curv_cochain_to_dict(c2),r2)
        self.assertEqual(self.C.curv_cochain_to_dict(c3),r3)
        self.assertEqual(self.C.curv_cochain_to_dict(c4),r4)
        self.assertEqual(self.C.curv_cochain_to_dict(c5),r5)