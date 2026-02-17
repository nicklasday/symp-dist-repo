import unittest
import sympy as sp
from symp_dist.algebra.tanaka_symbols.symplectic_symbol import SympSymb
from symp_dist.algebra.tanaka_symbols.tanaka_symbol_basis_elt import TSymbBasisElt
from symp_dist.utils.exceptions import invalid_parent_exception

class test_T_symb_basis_elt(unittest.TestCase):
    def setUp(self):
        self.T3=SympSymb(3)
        self.T5=SympSymb(5)
        self.C=self.T3.cochain_complex
        self.E=self.T3.ext_alg
        
        self.Y3=self.T3.basis[0]
        self.H3=self.T3.basis[1]
        self.E3=self.T3.basis[2]
        self.X3=self.T3.basis[3]
        self.e13=self.T3.basis[4]
        self.e23=self.T3.basis[5]
        self.N3=self.T3.basis[6]
        
        self.XY=self.T3.elt([1,0,0,1,0,0,0])
        self.Xe2=self.T3.elt([0,0,0,1,0,1,0])
        
        self.Y5=self.T5.basis[0]
        self.H5=self.T5.basis[1]
        self.E5=self.T5.basis[2]
        self.X5=self.T5.basis[3]
        self.e15=self.T5.basis[4]
        self.e25=self.T5.basis[5]
        self.e35=self.T5.basis[6]
        self.e45=self.T5.basis[7]
        self.N35=self.T5.basis[8]
        
    def test_init(self):
        # These are implicitly checking vec_rep attributes
        self.assertEqual(self.X3,self.T3.elt([0,0,0,1,0,0,0]))
        self.assertEqual(self.X3,TSymbBasisElt('X','X',-1,self.T3,3,sp.zeros(1)))

        self.assertEqual(self.e35,self.T5.elt([0,0,0,0,0,0,1,0,0]))
        self.assertEqual(self.e35,TSymbBasisElt('e_3','e_3',10,self.T5,6,sp.zeros(1)))
        
        self.assertEqual(self.X3.parent,self.T3)
        self.assertEqual(self.X3.str_rep,'X')
        self.assertEqual(str(self.X3),'X')

        # Maybe more testing here? Make sure I understand what I want this to do
        # At the moment, I'm just using it for sorting, though.
        self.assertLess(self.Y3,self.X3)
        self.assertRaises(invalid_parent_exception,self.X3.__le__,self.X5)
        
        self.assertEqual(self.Y3.wght,1)
        self.assertEqual(self.H3.wght,0)
        self.assertEqual(self.E3.wght,0)
        self.assertEqual(self.X3.wght,-1)
        self.assertEqual(self.e13.wght,-1)
        self.assertEqual(self.e23.wght,-2)
        self.assertEqual(self.N3.wght,-3)
        
        self.assertEqual(self.T3.heis_dim,3)
        self.assertEqual(self.Y3.parent,self.T3)
        
    def test_add_etc(self):
        self.assertEqual(self.X3+self.Y3,self.XY)
        self.assertEqual(self.X3+self.Y3+self.Y3,self.XY+self.Y3)
        self.assertEqual(self.X3-self.Y3+self.Y3,self.X3)
        self.assertEqual(-self.Y3+self.X3+self.Y3,self.X3)
        self.assertEqual(-self.Y3+2*self.X3+self.Y3-self.X3,self.X3)
        self.assertEqual(self.X3+self.T3.elt([0,0,0,0,0,1,0]),self.Xe2)
        self.assertEqual(-self.X3,self.T3.elt([0,0,0,-1,0,0,0]))
        self.assertEqual(self.X3*8-7*self.X3,self.X3)
        
    def test_ad(self):
        self.assertEqual(self.Y3.ad(self.Y3),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.Y3.ad(self.H3),self.T3.elt([2,0,0,0,0,0,0]))
        self.assertEqual(self.Y3.ad(self.E3),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.Y3.ad(self.X3),self.T3.elt([0,-1,0,0,0,0,0]))
        self.assertEqual(self.Y3.ad(self.e13),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.Y3.ad(self.e23),self.T3.elt([0,0,0,0,1,0,0]))
        self.assertEqual(self.Y3.ad(self.N3),self.T3.elt([0,0,0,0,0,0,0]))
        
        self.assertEqual(self.H3.ad(self.Y3),self.T3.elt([-2,0,0,0,0,0,0]))
        self.assertEqual(self.H3.ad(self.H3),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.H3.ad(self.E3),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.H3.ad(self.X3),self.T3.elt([0,0,0,2,0,0,0]))
        self.assertEqual(self.H3.ad(self.e13),self.T3.elt([0,0,0,0,-1,0,0]))
        self.assertEqual(self.H3.ad(self.e23),self.T3.elt([0,0,0,0,0,1,0]))
        self.assertEqual(self.H3.ad(self.N3),self.T3.elt([0,0,0,0,0,0,0]))
        
        self.assertEqual(self.E3.ad(self.Y3),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.E3.ad(self.H3),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.E3.ad(self.E3),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.E3.ad(self.X3),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.E3.ad(self.e13),self.T3.elt([0,0,0,0,1,0,0]))
        self.assertEqual(self.E3.ad(self.e23),self.T3.elt([0,0,0,0,0,1,0]))
        self.assertEqual(self.E3.ad(self.N3),self.T3.elt([0,0,0,0,0,0,2]))
        
        self.assertEqual(self.X3.ad(self.Y3),self.T3.elt([0,1,0,0,0,0,0]))
        self.assertEqual(self.X3.ad(self.H3),self.T3.elt([0,0,0,-2,0,0,0]))
        self.assertEqual(self.X3.ad(self.E3),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.X3.ad(self.X3),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.X3.ad(self.e13),self.T3.elt([0,0,0,0,0,1,0]))
        self.assertEqual(self.X3.ad(self.e23),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.X3.ad(self.N3),self.T3.elt([0,0,0,0,0,0,0]))
        
        self.assertEqual(self.e13.ad(self.Y3),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.e13.ad(self.H3),self.T3.elt([0,0,0,0,1,0,0]))
        self.assertEqual(self.e13.ad(self.E3),self.T3.elt([0,0,0,0,-1,0,0]))
        self.assertEqual(self.e13.ad(self.X3),self.T3.elt([0,0,0,0,0,-1,0]))
        self.assertEqual(self.e13.ad(self.e13),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.e13.ad(self.e23),self.T3.elt([0,0,0,0,0,0,-1]))
        self.assertEqual(self.e13.ad(self.N3),self.T3.elt([0,0,0,0,0,0,0]))
        
        self.assertEqual(self.e23.ad(self.Y3),self.T3.elt([0,0,0,0,-1,0,0]))
        self.assertEqual(self.e23.ad(self.H3),self.T3.elt([0,0,0,0,0,-1,0]))
        self.assertEqual(self.e23.ad(self.E3),self.T3.elt([0,0,0,0,0,-1,0]))
        self.assertEqual(self.e23.ad(self.X3),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.e23.ad(self.e13),self.T3.elt([0,0,0,0,0,0,1]))
        self.assertEqual(self.e23.ad(self.e23),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.e23.ad(self.N3),self.T3.elt([0,0,0,0,0,0,0]))
        
        self.assertEqual(self.N3.ad(self.Y3),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.N3.ad(self.H3),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.N3.ad(self.E3),self.T3.elt([0,0,0,0,0,0,-2]))
        self.assertEqual(self.N3.ad(self.X3),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.N3.ad(self.e13),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.N3.ad(self.e23),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.N3.ad(self.N3),self.T3.elt([0,0,0,0,0,0,0]))
            
    def test_dual_ad_cc(self):
        self.assertEqual(self.Y3.ad(self.Y3,mod='g_dual'),self.T3.elt([0,-2,0,0,0,0,0]))
        self.assertEqual(self.Y3.ad(self.H3,mod='g_dual'),self.T3.elt([0,0,0,1,0,0,0]))
        self.assertEqual(self.Y3.ad(self.E3,mod='g_dual'),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.Y3.ad(self.X3,mod='g_dual'),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.Y3.ad(self.e13,mod='g_dual'),self.T3.elt([0,0,0,0,0,-1,0]))
        self.assertEqual(self.Y3.ad(self.e23,mod='g_dual'),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.Y3.ad(self.N3,mod='g_dual'),self.T3.elt([0,0,0,0,0,0,0]))
        
        self.assertEqual(self.H3.ad(self.Y3,mod='g_dual'),self.T3.elt([2,0,0,0,0,0,0]))
        self.assertEqual(self.H3.ad(self.H3,mod='g_dual'),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.H3.ad(self.E3,mod='g_dual'),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.H3.ad(self.X3,mod='g_dual'),self.T3.elt([0,0,0,-2,0,0,0]))
        self.assertEqual(self.H3.ad(self.e13,mod='g_dual'),self.T3.elt([0,0,0,0,1,0,0]))
        self.assertEqual(self.H3.ad(self.e23,mod='g_dual'),self.T3.elt([0,0,0,0,0,-1,0]))
        self.assertEqual(self.H3.ad(self.N3,mod='g_dual'),self.T3.elt([0,0,0,0,0,0,0]))
        
        self.assertEqual(self.E3.ad(self.Y3,mod='g_dual'),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.E3.ad(self.H3,mod='g_dual'),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.E3.ad(self.E3,mod='g_dual'),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.E3.ad(self.X3,mod='g_dual'),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.E3.ad(self.e13,mod='g_dual'),self.T3.elt([0,0,0,0,-1,0,0]))
        self.assertEqual(self.E3.ad(self.e23,mod='g_dual'),self.T3.elt([0,0,0,0,0,-1,0]))
        self.assertEqual(self.E3.ad(self.N3,mod='g_dual'),self.T3.elt([0,0,0,0,0,0,-2]))
        
        self.assertEqual(self.X3.ad(self.Y3,mod='g_dual'),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.X3.ad(self.H3,mod='g_dual'),self.T3.elt([-1,0,0,0,0,0,0]))
        self.assertEqual(self.X3.ad(self.E3,mod='g_dual'),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.X3.ad(self.X3,mod='g_dual'),self.T3.elt([0,2,0,0,0,0,0]))
        self.assertEqual(self.X3.ad(self.e13,mod='g_dual'),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.X3.ad(self.e23,mod='g_dual'),self.T3.elt([0,0,0,0,-1,0,0]))
        self.assertEqual(self.X3.ad(self.N3,mod='g_dual'),self.T3.elt([0,0,0,0,0,0,0]))
        
        self.assertEqual(self.e13.ad(self.Y3,mod='g_dual'),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.e13.ad(self.H3,mod='g_dual'),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.e13.ad(self.E3,mod='g_dual'),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.e13.ad(self.X3,mod='g_dual'),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.e13.ad(self.e13,mod='g_dual'),self.T3.elt([0,-1,1,0,0,0,0]))
        self.assertEqual(self.e13.ad(self.e23,mod='g_dual'),self.T3.elt([0,0,0,1,0,0,0]))
        self.assertEqual(self.e13.ad(self.N3,mod='g_dual'),self.T3.elt([0,0,0,0,0,1,0]))
        
        self.assertEqual(self.e23.ad(self.Y3,mod='g_dual'),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.e23.ad(self.H3,mod='g_dual'),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.e23.ad(self.E3,mod='g_dual'),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.e23.ad(self.X3,mod='g_dual'),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.e23.ad(self.e13,mod='g_dual'),self.T3.elt([1,0,0,0,0,0,0]))
        self.assertEqual(self.e23.ad(self.e23,mod='g_dual'),self.T3.elt([0,1,1,0,0,0,0]))
        self.assertEqual(self.e23.ad(self.N3,mod='g_dual'),self.T3.elt([0,0,0,0,-1,0,0]))
        
        self.assertEqual(self.N3.ad(self.Y3,mod='g_dual'),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.N3.ad(self.H3,mod='g_dual'),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.N3.ad(self.E3,mod='g_dual'),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.N3.ad(self.X3,mod='g_dual'),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.N3.ad(self.e13,mod='g_dual'),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.N3.ad(self.e23,mod='g_dual'),self.T3.elt([0,0,0,0,0,0,0]))
        self.assertEqual(self.N3.ad(self.N3,mod='g_dual'),self.T3.elt([0,0,2,0,0,0,0]))
    
    # def test_ext_ad_cc(self):
    #     # In fact, E is only a g_+ module
    #     self.assertEqual(self.X3.ad(self.E.elt_from_cd({})),self.E.elt_from_cd())
    #     self.assertEqual(self.X3.ad(self.E.elt_from_cd()),self.E.elt_from_cd({}))
    #     self.assertEqual(self.T3.elt().ad(self.E.elt_from_cd({('X','e_2'):1})),self.E.elt_from_cd({}))
    #     self.assertEqual(self.T3.elt().ad(self.E.elt_from_cd({})),self.E.elt_from_cd({}))
        
    #     self.assertEqual(self.X3.ad(self.E.elt_from_cd({('X','e_1'):1})),self.E.elt_from_cd({}))
    #     self.assertEqual(self.X3.ad(self.E.elt_from_cd({('X','e_2'):2})),self.E.elt_from_cd({('X','e_1'):-2}))
    #     self.assertEqual(self.X3.ad(self.E.elt_from_cd({('X','N'):1})),self.E.elt_from_cd({}))

    #     self.assertEqual(self.X3.ad(self.E.elt_from_cd({('e_1','X'):1})),self.E.elt_from_cd({}))
    #     self.assertEqual(self.X3.ad(self.E.elt_from_cd({('N','e_2'):4})),self.E.elt_from_cd({('e_1','N'):4}))
    #     self.assertEqual(self.X3.ad(self.E.elt_from_cd({('e_1','N'):1})),self.E.elt_from_cd({}))

    #     self.assertEqual((self.X3+2*self.e13).ad(self.E.elt_from_cd()),self.E.elt_from_cd({}))
    #     self.assertEqual((self.X3+2*self.e13).ad(self.E.elt_from_cd({('e_2','X'):1})),
    #                      self.E.elt_from_cd({('X','e_1'):1}))
    #     self.assertEqual((self.X3+2*self.e13).ad(self.E.elt_from_cd({('e_2','e_1'):1})),
    #                      self.E.elt_from_cd({('X','e_1'):2}))
    #     self.assertEqual((self.X3+2*self.e13-5*self.N3).ad(self.E.elt_from_cd({('e_1','N'):1})),
    #                      self.E.elt_from_cd({('e_1','e_2'):2}))

    #     self.assertEqual(self.N3.ad(self.E.elt_from_cd({('N','X'):1})),self.E.elt_from_cd({}))
    #     self.assertEqual(self.N3.ad(self.E.elt_from_cd({('N','e_1'):1})),self.E.elt_from_cd({}))
    #     self.assertEqual(self.N3.ad(self.E.elt_from_cd({('N','e_2'):1})),self.E.elt_from_cd({}))