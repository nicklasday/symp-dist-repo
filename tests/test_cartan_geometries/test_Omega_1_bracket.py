import unittest
import sympy as sp
from symp_dist.algebra.tanaka_symbols.symplectic_symbol import SympSymb
from symp_dist.distributions import Free_distr,Vector_Field
from symp_dist.cartan_geometries import Geom_Prolongation

class test_Omega_1_bracket(unittest.TestCase):
    def setUp(self):
        self.K=sp.IndexedBase('K')
        self.T=SympSymb(5)
        self.C=self.T.cochain_complex
        self.D=Free_distr(self.T)[0]
        self.P=Geom_Prolongation(self.D,{})

    def test_Omega_1_bracket_on_section(self):
        B=self.P.alg.basis
        y,h,e=sp.symbols('y,h,e')
        hey_z={y:0,h:0,e:0}
        def tf1(i, j, min_w=-sp.oo):
            return self.P.Omega_1_bracket(B[i].vec,B[j].vec,min_w).subs(hey_z)
        
        for i in range(3,9): # Don't test all of them for the sake of time
            vi=Vector_Field(sp.Matrix([0]*(i)+[1]+[0]*(8-i)),self.D)
            for j in range(i,9):
                vj=Vector_Field(sp.Matrix([0]*(j)+[1]+[0]*(8-j)),self.D)
                self.assertEqual(tf1(i,j),sp.Matrix(self.D.bracket(vi,vj).vec))
                self.assertEqual(tf1(j,i),-sp.Matrix(self.D.bracket(vi,vj).vec))
                temp=sp.Matrix(self.D.bracket(vi,vj).vec)
                temp[8]=0
                self.assertEqual(tf1(i,j,min_w=-4),temp)
                self.assertEqual(tf1(j,i,min_w=-4),-temp)
                temp[7]=0
                self.assertEqual(tf1(i,j,min_w=-3),temp)
                self.assertEqual(tf1(j,i,min_w=-3),-temp)
                temp[6]=0
                self.assertEqual(tf1(i,j,min_w=-2),temp)
                self.assertEqual(tf1(j,i,min_w=-2),-temp)
                temp[5]=0
                self.assertEqual(tf1(i,j,min_w=-1),temp)
                self.assertEqual(tf1(j,i,min_w=-1),-temp)
                temp[4]=0
                temp[3]=0
                self.assertEqual(tf1(i,j,min_w=0),temp)
                self.assertEqual(tf1(j,i,min_w=0),temp)

        def tf2(X1, X2, min_w=-sp.oo):
            return self.P.Omega_1_bracket(X1,X2,min_w).subs(hey_z)
        VL=[]
        VL.append([0,0,0,0,0,0,0,0,0])
        VL.append([0,0,0,y,h,-2,0,0,0])
        VL.append([0,0,0,0,1,2,e**2,h,y])
        VL.append([0,0,0,0,2*self.K[-1],1,y*h,0,1])
        
        for A1 in VL:
            v1=Vector_Field(A1,self.D)
            for A2 in VL:
                v2=Vector_Field(A2,self.D)
                temp1=tf2(A1,A2)
                temp2=sp.Matrix(self.D.bracket(v1,v2).vec).subs({self.K[-1,4]:0,self.K[-1,5]:0}) # This is for a single case
                temp2=temp2.subs({y:0,h:0,e:0})
                self.assertEqual(temp1,temp2)

    def test_Omega_1_bracket_verticals(self):
        Y,H,E=self.T.basis[0:3]
        y,h,e=sp.symbols('y,h,e')
        VL=[]
        VL.append([0,0,0,0,0,0,0,0,0])
        VL.append([0,0,0,y,h,-2,0,0,0])
        VL.append([0,0,0,0,1,2,e**2,h,y])
        VL.append([0,0,0,0,2*self.K[-1],1,y*h,0,1])
        for X in [Y,H,E]:
            for V in VL:
                self.assertEqual(self.P.Omega_1_bracket(X.vec,V),X.ad(self.T.elt(V)).vec)
            for X2 in [Y,H,E]:
                self.assertEqual(self.P.Omega_1_bracket(X.vec,X2.vec),X.ad(X2).vec)
                self.assertEqual(self.P.Omega_1_bracket(X.vec,y*h**2*X2.vec),X.ad(y*h**2*X2).vec)
