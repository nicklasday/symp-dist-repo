import unittest
import sympy as sp
from symp_dist.utils import cochain_helpers as ch
from symp_dist.algebra.tanaka_symbols.symplectic_symbol import SympSymb
from symp_dist.distributions import Free_distr
from symp_dist.cartan_geometries import Geom_Prolongation

class test_Omega_1_curv(unittest.TestCase):
    # Test equivariance and agreement with base curvature along the section
    def setUp(self):
        self.K=sp.IndexedBase('K')
        self.T=SympSymb(5)
        self.C=self.T.cochain_complex
        self.D=Free_distr(self.T)[0]
        self.P=Geom_Prolongation(self.D,{})

    def test_Omega_1_curv_on_section(self):
        sd={sp.symbols('h'):0,sp.symbols('e'):0,sp.symbols('y'):0}
        z=self.P.Omega_1_curv.subs(sd)-self.D.curv
        ch.simplify_cochain(z)
        self.assertEqual(z,self.C.elt({}))
    
## ------------------------------------------------------------------------------------
# Just commented out for time purposes
    def test_Omega_1_curv_equivar(self):
        Y,H,E=self.T.basis[0:3]
        y,h,e=sp.symbols('y,h,e')
        y1,h1,e1=sp.symbols('y1,h1,e1')
        # We should have K(R_{p1}p) = Ad(p1^{-1})(K(p))
        Ad_p1_Kp=(-sp.exp(-2*h1)*y1*Y-h1*H-e1*E).Ad(self.P.Omega_1_curv,mod='CE')
        KR_p1_p=self.P.Omega_1_curv.subs({e:e+e1,h:h+h1,y:sp.exp(2*h1)*y+y1})
        result=Ad_p1_Kp-KR_p1_p
        ch.simplify_cochain(result)
        self.assertEqual(result,self.C.elt({}))
    