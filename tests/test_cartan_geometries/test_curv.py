import unittest
import sympy as sp
from src.utils import cochain_helpers as ch
from src.algebra.tanaka_symbols.symplectic_symbol import SympSymb
from src.distributions import Free_distr
from src.cartan_geometries import Geom_Prolongation

# Commented out for time purposes
class test_curv(unittest.TestCase):
    # To do: Write more tests here
    def setUp(self):
        self.K=sp.IndexedBase('K')
        self.T=SympSymb(5)
        self.C=self.T.cochain_complex
        self.D=Free_distr(self.T)[0]
        self.P=Geom_Prolongation(self.D,{})
                        
    def test_curv(self):
    # Checking curv using brackets
        F0=sp.eye(len(self.T.basis))
        F1=sp.eye(len(self.T.basis))
        F2=sp.eye(len(self.T.basis))

        F1[0,2]=1
        F1[1,3]=1
        F1[2,4]=1
        F1[5,6]=1
        F1[4,8]=1

        F2[3,5]=-5
        F2[3,6]=3
        F2[3,7]=1
        F2[3,8]=-1

        for i in range(len([F0,F1,F2])):
            F=[F0,F1,F2][i]
            FK=[self.P.curv(F,self.C.elt({}),0)]                
            for w in range(1,15):
                temp=sum(FK,start=self.C.elt({}))
                FK.append(self.P.curv(F,temp,w))
                FKw=sum(FK[0:w+1],start=self.C.elt({})) # weight including w
                # FKw_dict=self.C.curv_cochain_to_dict(FKw)
                for i in range(3,len(self.T.basis)):
                    for j in range(i+1,len(self.T.basis)):
                        Xij=F.inv()*(self.P.bracket(F.col(i),F.col(j)))-self.T.basis[i].ad(self.T.basis[j]).vec
                        eij=self.T.ext_alg.elt_from_cd({(self.T.basis_strs[i],self.T.basis_strs[j]):1})
                        Yij=FKw.apply_cochain_map(eij).vec
                        for k in range(len(self.T.basis)):
                            if self.T.basis[k].wght>w+self.T.basis[i].wght+self.T.basis[j].wght: Xij[k]=0
                        self.assertEqual(Xij,Yij)

    def test_equiv_curv(self):
    # Check that an equivariant frame gives equivariant curvature
        F0=sp.eye(len(self.T.basis))
        F1=sp.eye(len(self.T.basis))
        F2=sp.eye(len(self.T.basis))

        F1[0,2]=1
        F1[1,3]=1
        F1[2,4]=1
        F1[5,6]=1
        F1[4,8]=1

        F2[3,5]=-5
        F2[3,6]=3
        F2[3,7]=1
        F2[3,8]=-1

        y,h,e=sp.symbols('y,h,e')
        g_mat=sp.exp(h*self.T.basis[1].ad_mat()+e*self.T.basis[2].ad_mat())*sp.exp(y*self.T.basis[0].ad_mat())
        g_mat_inv=sp.simplify(g_mat.inv())
        g_inv=-y*sp.exp(-2*h)*self.T.basis[0]-h*self.T.basis[1]-e*self.T.basis[2]

        gF0=g_mat_inv*F0*g_mat
        gF1=g_mat_inv*F1*g_mat
        gF2=g_mat_inv*F2*g_mat

        gF_list=[gF0,gF1,gF2]

        for i in range(len(gF_list)):
            gF=gF_list[i]
            gFK=[self.P.curv(gF,self.C.elt({}),0)]                
            for w in range(1,15):
                temp=sum(gFK,start=self.C.elt({}))
                gFK.append(self.P.curv(gF,temp,w))
            total_K=sum(gFK,start=self.C.elt({})) # should be equivariant
            # gFKw=sum(gFK[0:w+1],start=self.C.elt({})) # weight including w
            total_K_0=total_K.subs({y:0,h:0,e:0})
            r=g_inv.Ad(total_K_0,mod='CE')-total_K
            ch.simplify_cochain(r)
            self.assertEqual(r,self.C.elt({}))