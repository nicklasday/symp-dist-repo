import unittest
import sympy as sp
from src.algebra.tanaka_symbols.symplectic_symbol import SympSymb

class iprod_test(unittest.TestCase):
    def setUp(self):
        self.T3=SympSymb(3)
        self.T5=SympSymb(5)
        self.T7=SympSymb(7)
        self.T9=SympSymb(9)
        self.C3=self.T3.cochain_complex
        self.C5=self.T5.cochain_complex
        self.C7=self.T7.cochain_complex
        self.C9=self.T9.cochain_complex
#         self.startTime = time.time()
        
#     def tearDown(self):
#         t = time.time() - self.startTime
#         print('%s: %.3f' % (self.id(), t))
        
        
    def test_T_symb_iprod(self):
        Y,H,E,X,e1,e2,e3,e4,e5,e6,e7,e8,N=self.T9.basis
        lens_list=[1,2,2,1]+[sp.factorial(i-1)/sp.factorial(2*4-i) for i in range(1,9)]+[1]
        
        
        self.assertEqual([A.iprod(A) for A in self.T9.basis],lens_list)
        for A in self.T9.basis:
            for B in self.T9.basis:
                if A!=B: self.assertEqual(A.iprod(B),0)
                    
        self.assertEqual((4*X+Y).iprod(X-5*Y),4*X.iprod(X)-5*Y.iprod(Y))
        self.assertEqual((3*e1+e2+e3).iprod(e1),3*e1.iprod(e1))
        self.assertEqual(e1.iprod(3*e1+e2+e3),3*e1.iprod(e1))
        self.assertEqual((3*e1+2*e2+e3).iprod(2*e1-e2+X),6*e1.iprod(e1)-2*e2.iprod(e2))
        
        
    def test_cochain_iprod(self):
        Y,H,E,X,e1,e2,e3,e4,e5,e6,e7,e8,N=self.T9.basis
        c1=self.C9.elt_from_cd({('X','N','E'):1})
        c2=self.C9.elt_from_cd({('e_1','N','X'):1})
        c3=self.C9.elt_from_cd({('e_2','e_1','e_2'):1})
        c4=self.C9.elt_from_cd({('e_3','e_4','N'):1})
        
        self.assertEqual(c1.iprod(c1),E.iprod(E)/(X.iprod(X)*N.iprod(N)))
        self.assertEqual(c2.iprod(c1),0)
        self.assertEqual(c2.iprod(c2),X.iprod(X)/(e1.iprod(e1)*N.iprod(N)))
        self.assertEqual(c3.iprod(c3),e2.iprod(e2)/(e2.iprod(e2)*e1.iprod(e1)))
        self.assertEqual(c4.iprod(c4),N.iprod(N)/(e3.iprod(e3)*e4.iprod(e4)))
        self.assertEqual(c1.iprod(3*c1),3*c1.iprod(c1))
        self.assertEqual(c1.iprod(c1+3*c2),c1.iprod(c1))
        self.assertEqual((c1-6*c4).iprod(3*c1+c3+c4),3*c1.iprod(c1)-6*c4.iprod(c4))

# # Commented out for time purposes
#     def test_im_proj(self):
#         # To do
#         # This test isn't valid; rewrite. For any 1 cochain, the its coboundary should project to itself.
#         # For any 2-cochain c, proj(c)=(proj^2(c), and (c-proj(c)) should be ortho. to c.
        
#         # assuming the image is correct...
#         Y,H,E,X,e1,e2,e3,e4,e5,e6,e7,e8,N=self.T9.basis
#         # Anything not from g_\wedge g_ \otimes g will project to 0
        
#         # deg 2, wght -1
#         K1,K2,K3=symbols('K1,K2,K3')
#         c1=self.C9.cochain({('Y','H','E'):1})
#         # from the image
#         c2=self.C9.cochain({('e_1','e7','N'):K2**2*K3+1,('X','e_1','e_3'):K2**2*K3+1})
#         c3=self.C9.cochain({('X','e_2','e_4'):K1+K2,('e_3','e_5','N'):-K1-K2,('X','e_3','e_5'):-K1-K2})
        
#         # deg 2, wght -2
#         c4=self.C9.cochain({('e_1','e_5','N'):1,('X','e_1','e_5'):1})
#         c5=self.C9.cochain({('X','e_2','e_6'):K1,('e_1','e_5','N'):K1,('e_2','e_4','N'):-K1})
        
        
#         self.assertEqual(self.C9.subspace_proj(c1,'im')[0],
#                          self.C9.subspace_proj(self.C9.subspace_proj(c1,'im')[0],'im')[0])
#         self.assertEqual(self.C9.subspace_proj(c2,'im')[0],c2)
#         self.assertEqual(self.C9.subspace_proj(c3,'im')[0],c3)
#         self.assertEqual(self.C9.subspace_proj(2*c2-c3,'im')[0],2*c2-c3)
        
        
#         self.assertEqual(self.C9.subspace_proj(c4,'im')[0],c4)
#         self.assertEqual(self.C9.subspace_proj(c5,'im')[0],c5)
#         self.assertEqual(self.C9.subspace_proj(c4+3*c2,'im')[0],c4+3*c2)