import unittest
import sympy as sp
from symp_dist.utils import lin_alg_helpers as lh
from symp_dist.algebra.tanaka_symbols.symplectic_symbol import SympSymb
from symp_dist.distributions import Free_distr,Vector_Field
from symp_dist.cartan_geometries import Geom_Prolongation


class test_bracket(unittest.TestCase):
    # I only need to check the Leibniz rule, Omega_1 curvature, and directional derivatives.
    # If these are correct, then P.bracket is correct
    def setUp(self):
        self.K=sp.IndexedBase('K')
        self.T=SympSymb(5)
        self.C=self.T.cochain_complex
        self.D=Free_distr(self.T)[0]
        self.P=Geom_Prolongation(self.D,{})


    def test_bracket_along_section(self):
        # extraneous test
        # Any vector fields tangent to the section should have brackets which are just lifts from the base
        y,h,e=sp.symbols('y,h,e')
        Y,H,E,X,e1,e2,e3,e4,N=[A.vec for A in self.T.basis]
        K0_Mat=lh.ind_diag(*[self.K[0]]*len(self.T.basis))
        K1_Mat=lh.ind_diag(*[self.K[1]]*len(self.T.basis))

        v1=X+K0_Mat*e1
        v2=e2-4*e3
        v3=e3+K1_Mat**2*N

        for u1 in [v1,v2,v3]:
            for u2 in [v1,v2,v3]:
                uvf1=Vector_Field(u1,self.D)
                uvf2=Vector_Field(u2,self.D)
                self.assertEqual((self.P.bracket(u1,u2).subs({y:0,h:0,e:0})),self.D.bracket(uvf1,uvf2).vec)

    def test_bracket_Leib(self):
        # Makes sure [U,f*V] = f[U,V] + U(f)*V
        y,h,e=sp.symbols('y,h,e')
        K0_Mat=lh.ind_diag(*[self.K[0]]*len(self.T.basis))
        
        B=[A.vec for A in self.T.basis]
        for b in B[1:5]:
            for V in [A.vec for A in self.T.basis]:
                self.assertEqual(self.P.bracket(V,y*b),y*self.P.bracket(V,b)+self.P.dir_der(y,V)*b)
                self.assertEqual(self.P.bracket(V,e*b),e*self.P.bracket(V,b)+self.P.dir_der(e,V)*b)
                self.assertEqual(self.P.bracket(V,h*b),h*self.P.bracket(V,b)+self.P.dir_der(h,V)*b)
                r=sp.zeros(len(b))
                for i in range(len(b)):
                    r[i,i]=self.P.dir_der(self.K[0],V)
                self.assertEqual(self.P.bracket(V,K0_Mat*b),K0_Mat*self.P.bracket(V,b)+r*b)

    def test_bracket_on_base(self):
        # Based on the correctness of Omega_1_bracket
        s_basis=[sp.eye(len(self.T.basis)).col(i) for i in range(len(self.T.basis))]
        for i in range(len(s_basis)):
            ei=s_basis[i]
            for j in range(len(s_basis)):
                ej=s_basis[j]
                for k in range(10):
                    self.assertEqual(self.P.bracket(ei,ej,k),self.P.Omega_1_bracket(ei,ej,k))