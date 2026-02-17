import unittest
import sympy as sp
from symp_dist.algebra.tanaka_symbols.symplectic_symbol import SympSymb
from symp_dist.distributions import Free_distr
from symp_dist.cartan_geometries import Geom_Prolongation

# Commented out for time purposes
class TestTensorAction(unittest.TestCase):
    def setUp(self):
        self.K=sp.IndexedBase('K')
        self.T=SympSymb(5)
        self.C=self.T.cochain_complex
        self.D=Free_distr(self.T)[0]
        self.P=Geom_Prolongation(self.D,{})

    def test_ad_on_basis(self):
        c0=self.C.elt({})
        c1=self.C.elt_from_cd({('X','e_1','X'):1})
        c2=self.C.elt_from_cd({('X','e_2','Y'):1})
        c3=self.C.elt_from_cd({('e_1','e_2','H'):1})
        c4=self.C.elt_from_cd({('N','e_2','N'):1})
        c5=self.C.elt_from_cd({('e_1','e_2','E'):1})

        Y0=self.C.elt({})
        Y1=self.C.elt_from_cd({('X','e_2','X'):-3,('X','e_1','H'):-1})
        Y2=self.C.elt_from_cd({('X','e_3','Y'):-4})
        Y3=self.C.elt_from_cd({('e_1','e_3','H'):-4,('e_1','e_2','Y'):2})
        Y4=self.C.elt_from_cd({('N','e_3','N'):-4})
        Y5=self.C.elt_from_cd({('e_1','e_3','E'):-4})

        H0=self.C.elt({})
        H1=self.C.elt_from_cd({('X','e_1','X'):3})
        H2=self.C.elt_from_cd({('X','e_2','Y'):-3})
        H3=self.C.elt_from_cd({('e_1','e_2','H'):4})
        H4=self.C.elt_from_cd({('N','e_2','N'):1})
        H5=self.C.elt_from_cd({('e_1','e_2','E'):4})

        E0=-c0
        E1=-c1
        E2=-c2
        E3=-2*c3
        E4=-c4
        E5=-2*c5

        c_list=[c0,c1,c2,c3,c4,c5]
        Y_list=[Y0,Y1,Y2,Y3,Y4,Y5]
        H_list=[H0,H1,H2,H3,H4,H5]
        E_list=[E0,E1,E2,E3,E4,E5]

        for p in [(self.T.basis[0],Y_list),(self.T.basis[1],H_list),(self.T.basis[2],E_list)]:
            for i in range(len(c_list)):
                self.assertEqual(p[0].ad(c_list[i],mod='CE'),p[1][i])

    def test_ad_on_lin_combs(self):
        Y,H,E=self.T.basis[0:3]
        c0=self.C.elt({})
        c1=self.C.elt_from_cd({('X','e_1','X'):1})
        c2=self.C.elt_from_cd({('X','e_2','Y'):1})
        c3=self.C.elt_from_cd({('e_1','e_2','H'):1})
        c4=self.C.elt_from_cd({('N','e_2','N'):1})
        c5=self.C.elt_from_cd({('e_1','e_2','E'):1})

        self.C.elt({})
        Y1=self.C.elt_from_cd({('X','e_2','X'):-3,('X','e_1','H'):-1})
        Y2=self.C.elt_from_cd({('X','e_3','Y'):-4})
        Y3=self.C.elt_from_cd({('e_1','e_3','H'):-4,('e_1','e_2','Y'):2})
        Y4=self.C.elt_from_cd({('N','e_3','N'):-4})
        self.C.elt_from_cd({('e_1','e_3','E'):-4})

        self.C.elt({})
        H1=self.C.elt_from_cd({('X','e_1','X'):3})
        H2=self.C.elt_from_cd({('X','e_2','Y'):-3})
        H3=self.C.elt_from_cd({('e_1','e_2','H'):4})
        H4=self.C.elt_from_cd({('N','e_2','N'):1})
        H5=self.C.elt_from_cd({('e_1','e_2','E'):4})

        E1=-c1
        E2=-c2
        E3=-2*c3
        E4=-c4
        E5=-2*c5

        self.assertEqual((Y+H).ad(c0,mod='CE'),c0)
        self.assertEqual((Y+3*E).ad(c1,mod='CE'),Y1+3*E1)
        self.assertEqual((2*E-4*H).ad(c2,mod='CE'),2*E2-4*H2)
        self.assertEqual((2*E+8*Y).ad(c3,mod='CE'),2*E3+8*Y3)

        self.assertEqual(Y.ad(c1+c2,mod='CE'),Y1+Y2)
        self.assertEqual(Y.ad(c1+3*c3,mod='CE'),Y1+3*Y3)
        self.assertEqual(H.ad(-2*c2+c3-c4,mod='CE'),-2*H2+H3-H4)
        self.assertEqual(E.ad(c4-5*c5,mod='CE'),E4-5*E5)

        self.assertEqual((Y+3*H).ad(c1-c2,mod='CE'),Y1-Y2+3*H1-3*H2)
        self.assertEqual((3*E-H).ad(c3-c4+3*c5,mod='CE'),3*E3-3*E4+9*E5-H3+H4-3*H5)
        self.assertEqual((H+E-2*Y).ad(c1-c2+3*c4,mod='CE'),H1-H2+3*H4+E1-E2+3*E4-2*Y1+2*Y2-6*Y4)

    def test_Ad_G0(self):
        """Checks Ad(G0)"""
        H,E=self.T.basis[1:3]
        c0=self.C.elt({})
        c1=self.C.elt_from_cd({('X','e_1','X'):1})
        c2=self.C.elt_from_cd({('X','e_2','Y'):1})
        c3=self.C.elt_from_cd({('e_1','e_2','H'):1})
        c4=self.C.elt_from_cd({('N','e_2','N'):1})
        c5=self.C.elt_from_cd({('e_1','e_2','E'):1})
        c_list=[c0,c1,c2,c3,c4,c5]
        
        h0=0
        h1=3
        h2=-3
        h3=4
        h4=1
        h5=4
        h_list=[h0,h1,h2,h3,h4,h5]
        
        e0=0
        e1=-1
        e2=-1
        e3=-2
        e4=-1
        e5=-2
        e_list=[e0,e1,e2,e3,e4,e5]
        
        for i in range(6):
            self.assertEqual(H.Ad(c_list[i],mod='CE'),sp.exp(h_list[i])*c_list[i])
            self.assertEqual(E.Ad(c_list[i],mod='CE'),sp.exp(e_list[i])*c_list[i])

        self.assertEqual((H+3*E).Ad(c1,mod='CE'),sp.exp(h1+3*e1)*c1)
        self.assertEqual((2*H-4*E).Ad(c1,mod='CE'),sp.exp(2*h1-4*e1)*c1)
        self.assertEqual((H+3*E).Ad(c2,mod='CE'),sp.exp(h2+3*e2)*c2)
        self.assertEqual((2*H-4*E).Ad(c2,mod='CE'),sp.exp(2*h2-4*e2)*c2)

    def test_ad_Gplus(self):
        """Tests that Ad(Y) agrees with exp(ad(Y))"""
        Y=self.T.basis[0]
        c0=self.C.elt({})
        c1=self.C.elt_from_cd({('X','e_1','X'):1})
        c2=self.C.elt_from_cd({('X','e_2','Y'):1})
        c3=self.C.elt_from_cd({('e_1','e_2','H'):1})
        c4=self.C.elt_from_cd({('N','e_2','N'):1})
        c5=self.C.elt_from_cd({('e_1','e_2','E'):1})
        c_list=[c0,c1,c2,c3,c4,c5]
        
        for c in c_list:
            # Compute r=Ad(Y)(c) by exponentiating "manually"
            r=c
            Yk=c
            for k in range(1,9):
                Yk=Y.ad(Yk,mod='CE')
                r+=Yk*sp.Rational(1,sp.factorial(k))
            if Y.Ad(c,mod='CE')!=r:
                print('c =',c)
                print('Y.Ad(c) =',Y.Ad(c,mod='CE'))
                print('r =',r)
            self.assertEqual(Y.Ad(c,mod='CE'),r)
            
    def TestAdInverses(self):
        """Tests that Ad(p)Ad(p^{-1})=Ad(p^{-1})Ad(p)=Id"""
        Y,H,E=self.T.basis[0:3]
        c0=self.C.elt({})
        c1=self.C.elt_from_cd({('X','e_1','X'):1})
        c2=self.C.elt_from_cd({('X','e_2','Y'):1})
        c3=self.C.elt_from_cd({('e_1','e_2','H'):1})
        c4=self.C.elt_from_cd({('N','e_2','N'):1})
        c5=self.C.elt_from_cd({('e_1','e_2','E'):1})
        y,h,e=sp.symbols('y,h,e')

        c_list=[c0,c1,c2,c3,c4,c5]
        p=y*Y+h*H+e*E
        p_inv=self.T.second_kind_inv(p)#-y*sp.exp(-2*h)*Y-h*H-e*E
        for c in c_list: 
            self.assertEqual(p.Ad(p_inv.Ad(c,mod='CE')),c)
            self.assertEqual(p_inv.Ad(p.Ad(c,mod='CE')),c)

    def test_Ad_left_action(self):
        """This is checking the coordinate expression for the G_+ multiplication,
        which determines the (right) principal action on the fibers"""
        Y,H,E=self.T.basis[0:3]
        c0=self.C.elt({})
        c1=self.C.elt_from_cd({('X','e_1','X'):1})
        c2=self.C.elt_from_cd({('X','e_2','Y'):1})
        c3=self.C.elt_from_cd({('e_1','e_2','H'):1})
        c4=self.C.elt_from_cd({('N','e_2','N'):1})
        c5=self.C.elt_from_cd({('e_1','e_2','E'):1})
        y1,h1,e1,y2,h2,e2=sp.symbols('y1,h1,e1,y2,h2,e2')
        

        c_list=[c0,c1,c2,c3,c4,c5]
        p1=y1*Y+h1*H+e1*E
        p2=y2*Y+h2*H+e2*E
        p12=self.T.second_kind_mul(p1,p2)#(sp.exp(2*h1)*y2+y1)*Y+(h1+h2)*H+(e1+e2)*E # Represents R_{p1}(p2)=p2*p1
        
        for c in c_list:
            self.assertEqual(p1.Ad(p2.Ad(c,mod='CE')),p12.Ad(c,mod='CE'))

    def test_Ad_via_Leibniz(self):
        """Checks Ad using the Leibniz rule for cochains"""
        B=self.T.basis
        v_list=[B[0],B[0]+B[1],B[2]-4*B[1]]
        v=sp.symbols('y')*B[0]+sp.symbols('h')*B[1]+sp.symbols('e')*B[2]
        
        c0=self.C.elt({})
        c1=self.C.elt_from_cd({('X','e_1','X'):1})
        c2=self.C.elt_from_cd({('X','e_2','Y'):1})
        c3=self.C.elt_from_cd({('e_1','e_2','H'):1})
        c4=self.C.elt_from_cd({('N','e_2','N'):1})
        c5=self.C.elt_from_cd({('e_1','e_2','E'):1})

        c_list=[c0,c1,c2,c3,c4,c5]
        for v in v_list:
            vi=self.T.second_kind_inv(v)
            for c in c_list:
                for i in range(3,len(B)):
                    X1=B[i]
                    w1=self.T.elt([0]*3+(vi).Ad(X1).vec[3:len(B)]).cast_as_ext_elt()
                    for j in range(i,len(B)):
                        X2=B[j]
                        w2=self.T.elt([0]*3+(vi).Ad(X2).vec[3:len(B)]).cast_as_ext_elt()
                        w=w1.wedge(w2)
                        test1=v.Ad(c.apply_cochain_map(w))
                        test2=v.Ad(c,mod='CE').apply_cochain_map_base(X1,X2)
                        test3=sp.simplify((test1-test2).vec)
                        if test3!=self.T.elt().vec:
                            print('v =',v)
                            print('c =',c)
                            print('X1 =',X1)
                            print('X2 =',X2)
                        self.assertEqual(sp.simplify((test1-test2).vec),self.T.elt().vec)