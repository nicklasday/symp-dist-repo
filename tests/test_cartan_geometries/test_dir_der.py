import unittest
import sympy as sp
from symp_dist.utils import lin_alg_helpers as lh
from symp_dist.algebra.tanaka_symbols.symplectic_symbol import SympSymb
from symp_dist.distributions import Free_distr
from symp_dist.cartan_geometries import Geom_Prolongation


class test_dir_der(unittest.TestCase):
    def setUp(self):
        self.K=sp.IndexedBase('K')
        self.T=SympSymb(5)
        self.C=self.T.cochain_complex
        self.D=Free_distr(self.T)[0]
        self.P=Geom_Prolongation(self.D,{})

    def test_dir_der_linear(self):
        # Check correctness on linear functions
        from sympy import symbols
        y,h,e=symbols('y,h,e')
        Y,H,E,X,e1,e2,e3,e4,N=[A.vec for A in self.T.basis]

        self.assertEqual(self.P.dir_der(y,Y,normal=False),1)
        self.assertEqual(self.P.dir_der(h,Y,normal=False),0)
        self.assertEqual(self.P.dir_der(e,Y,normal=False),0)
        self.assertEqual(self.P.dir_der(self.K[0],Y,normal=False),0)

        self.assertEqual(self.P.dir_der(y,H,normal=False),2*y)
        self.assertEqual(self.P.dir_der(h,H,normal=False),1)
        self.assertEqual(self.P.dir_der(e,H,normal=False),0)
        self.assertEqual(self.P.dir_der(self.K[0],H,normal=False),0)

        self.assertEqual(self.P.dir_der(y,E,normal=False),0)
        self.assertEqual(self.P.dir_der(h,E,normal=False),0)
        self.assertEqual(self.P.dir_der(e,E,normal=False),1)
        self.assertEqual(self.P.dir_der(self.K[0],E,normal=False),0)

        self.assertEqual(self.P.dir_der(y,X,normal=False),-y**2)
        self.assertEqual(self.P.dir_der(h,X,normal=False),-y)
        self.assertEqual(self.P.dir_der(e,X,normal=False),0)
        self.assertEqual(self.P.dir_der(self.K[0],X,normal=False),sp.exp(2*h)*self.K[0,3])

        self.assertEqual(self.P.dir_der(y,e1,normal=False),0)
        self.assertEqual(self.P.dir_der(h,e1,normal=False),0)
        self.assertEqual(self.P.dir_der(e,e1,normal=False),0)
        self.assertEqual(self.P.dir_der(self.K[0],e1,normal=False),sp.exp(e-3*h)*self.K[0,4])

        self.assertEqual(self.P.dir_der(y,e2,normal=False),0)
        self.assertEqual(self.P.dir_der(h,e2,normal=False),0)
        self.assertEqual(self.P.dir_der(e,e2,normal=False),0)
        self.assertEqual(self.P.dir_der(self.K[0],e2,normal=False),
                         3*y*sp.exp(e-3*h)*self.K[0,4]+sp.exp(e-h)*self.K[0,5])
        
        
        p=y*self.T.basis[0]+h*self.T.basis[1]+e*self.T.basis[2]
        temp=p.Ad(self.T.basis[5])
        r=sum([temp.vec[i]*self.D.abn_ind_der(self.K[0,0,0],i) for i in range(3,len(temp.vec))])
        self.assertEqual(sp.simplify(self.P.dir_der(self.K[0,0,0],e2,normal=False)-r),0)

        self.assertEqual(self.P.dir_der(y,e3,normal=False),0)
        self.assertEqual(self.P.dir_der(h,e3,normal=False),0)
        self.assertEqual(self.P.dir_der(e,e3,normal=False),0)
        self.assertEqual(self.P.dir_der(self.K[0],e3,normal=False),
                         6*y**2*sp.exp(e-3*h)*self.K[0,4]+4*y*sp.exp(e-h)*self.K[0,5]+sp.exp(e+h)*self.K[0,6])

    def test_dir_der_Leib(self):
        y,h,e=sp.symbols('y,h,e')
        Y,H,E,X,e1,e2,e3,e4,N=[A.vec for A in self.T.basis]
        K_Mat=lh.ind_diag(*[self.K[-1]]*len(self.T.basis))
        
        f1=y*h**2*e
        f2=y*self.K[0]
        f3=self.K[0]**2+1

        v1=Y+2*h*H
        v2=2*Y+K_Mat*X
        v3=K_Mat*H+2*y*e1

        # Check linearization in each argument
        # I could do more here, but it's probably right
        for f in [y,h,e,self.K[0]]:
            self.assertEqual(self.P.dir_der(f,v1,normal=False),self.P.dir_der(f,Y,normal=False)+2*h*self.P.dir_der(f,H,normal=False))
            self.assertEqual(self.P.dir_der(f,v2,normal=False),2*self.P.dir_der(f,Y,normal=False)+self.K[-1]*self.P.dir_der(f,X,normal=False))
            self.assertEqual(self.P.dir_der(f,v3,normal=False),self.K[-1]*self.P.dir_der(f,H,normal=False)+2*y*self.P.dir_der(f,e1,normal=False))

        for A in [Y,H,E,X,e1,e2,e3,e4,N]:
            self.assertEqual(self.P.dir_der(f1,A,normal=False),
                             self.P.dir_der(y,A,normal=False)*h**2*e+y*2*h*self.P.dir_der(h,A,normal=False)*e
                             +y*h**2*self.P.dir_der(e,A,normal=False))
            self.assertEqual(sp.simplify(self.P.dir_der(f2,A,normal=False)-(
                             self.P.dir_der(y,A,normal=False)*self.K[0]+y*self.P.dir_der(self.K[0],A,normal=False))),0)
            self.assertEqual(sp.simplify(self.P.dir_der(f3,A,normal=False)-(
                             2*self.K[0]*self.P.dir_der(self.K[0],A,normal=False))),0)