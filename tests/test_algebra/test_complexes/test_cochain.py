import unittest
import sympy as sp
from src.algebra.tanaka_symbols.symplectic_symbol import SympSymb

# # Temporarily commented out for time purposes

class TestCochainComplex(unittest.TestCase):
    def setUp(self):
        self.T3=SympSymb(3)
        self.C3=self.T3.cochain_complex
        self.T5=SympSymb(5)
        self.C5=self.T5.cochain_complex
        self.T7=SympSymb(7)
        self.C7=self.T7.cochain_complex
        
#         self.startTime = time.time()
        
#     def tearDown(self):
#         t = time.time() - self.startTime
#         print('%s: %.3f' % (self.id(), t))
        
#     def test_init(self):
#         print(vars(self.C3))
#         print(vars(self.C3.ext_alg))

#     def test_init(self):
#         # Just choose some random cochains 
#         # and make sure constructors agree
        
#         c0=self.C7.elt_from_cd()
#         c1=self.C7.elt_from_cd({('X','e_3','Y'):-2,('e_1','e_2','N'):-4,('e_2','e_5','e_4'):-3})
#         c2=self.C7.elt_from_cd({('e_2','e_1','Y'):-1,('e_6','e_2','N'):10,('e_1','N','e_4'):2})
#         c3=c1+c2
#         c_list=[c0,c1,c2,c3]
        
#         for c in c_list: c.find_mat_rep()
#         d_list=[self.C7.cochain(c.mat_rep,deg=2,mat_rep=True) for c in c_list]
        
#         for i in range(len(c_list)):
#             self.assertEqual(c_list[i],d_list[i])
            
#         c4=self.C5.cochain({('e_1','e_2','e_3','Y'):1})
#         c5=self.C5.cochain({('e_1','e_2','e_3','Y'):1,('e_1','e_2','N','N'):-4,('e_2','e_3','e_4','E'):-3})
#         c6=c4+c5
        
#         c_list2=[c4,c5,c6]
        
#         for c in c_list2: c.find_mat_rep()
#         d_list2=[self.C5.cochain(c.mat_rep,deg=3,mat_rep=True) for c in c_list2]
        
#         for i in range(len(c_list2)):
#             self.assertEqual(c_list2[i],d_list2[i])
        
    def test_wedge_add_mul(self):
        r0=self.T3.ext_alg.elt_from_cd({})
        r1=self.T3.ext_alg.elt_from_cd({tuple():1})
        r2=self.T3.ext_alg.elt_from_cd({('e_2','X'):1})
        r3=self.T3.ext_alg.elt_from_cd({('e_2',):1})
        r4=self.T3.ext_alg.elt_from_cd({('e_1',):1})
        
        s0=self.T5.ext_alg.elt_from_cd({})
        s1=self.T5.ext_alg.elt_from_cd({tuple():3})
        s2=self.T5.ext_alg.elt_from_cd({('e_2',):2,('e_1',):-1,('X','N'):1})
        s3=self.T5.ext_alg.elt_from_cd({('X','N'):-3,('e_2','N'):2})
        s4=self.T5.ext_alg.elt_from_cd({('X','e_2','N'):6,('X','e_1','N'):-3,('e_1','e_2','N'):-2})
        c0=self.C5.elt_from_cd({})
        c1=self.C5.elt_from_cd({('Y',):1}) # e2|->Y
        c2=self.C5.elt_from_cd({('X','Y'):-3})
        c3=self.C5.elt_from_cd({('X','H'):1,('N','e_3'):-3})
        
        t0=self.T5.elt()
        t1=self.T5.elt([1]+[0]*(len(self.T5.basis)-1))
        t2=self.T5.elt([-1]+[0]*(len(self.T5.basis)-1))
        t3=self.T5.elt([0]*(len(self.T5.basis)-1)+[1])
        t4=self.T5.elt([1,-2]+[0]*(len(self.T5.basis)-2))
        
        b1=self.T5.basis[0]
        b2=self.T5.basis[1]
        
        # __add__
        
        self.assertEqual(r0+r1,r1)
        self.assertEqual(r2+r0,r2)
        self.assertEqual(r2+r4,self.T3.ext_alg.elt_from_cd({('X','e_2'):-1,('e_1',):1}))
        self.assertEqual(r3+r3,2*r3)
        self.assertEqual(r3+r3,self.T3.ext_alg.elt_from_cd({('e_2',):2}))
        self.assertEqual(r3-r3,r0)
        self.assertEqual(-r3,self.T3.ext_alg.elt_from_cd({})-self.T3.ext_alg.elt_from_cd({('e_2',):1}))
        self.assertEqual(-r3,self.T3.ext_alg.elt_from_cd({})+self.T3.ext_alg.elt_from_cd({('e_2',):-1}))
        
        self.assertEqual(c0+c1,c1)
        self.assertEqual(c0-c1,-c1)
        self.assertEqual(c0-c1,self.C5.elt_from_cd({('Y',):-1}))
        self.assertEqual(c1+2*c2,self.C5.elt_from_cd({('Y',):1,('X','Y'):-6}))
        self.assertEqual(4*c2,c2+c2+c2+c2)
        self.assertEqual(4*(c1+c2),self.C5.elt_from_cd({('Y',):4,('X','Y'):-12}))
        
        self.assertEqual(b1,t1)
        self.assertEqual(b1,b1+t0)
        self.assertEqual(t1,b1+t0)
        self.assertEqual(t1,t1+t0)
        self.assertEqual(t1,t0+b1)
        self.assertEqual(t1,t0+t1)
        
        self.assertEqual(t0-t1,t2)
        self.assertEqual(-t1+t0,t2)
        self.assertEqual(t3+t3,self.T5.elt([0]*(len(self.T5.basis)-1)+[2]))
        self.assertEqual(b1-2*b2,t4)
        
        
        # __mul__
        self.assertEqual(3*r0,r0)
        self.assertEqual(0*r2,r0)
        self.assertEqual(-3*r1,self.T3.ext_alg.elt_from_cd({tuple():-3}))
        self.assertEqual(3*r2,self.T3.ext_alg.elt_from_cd({('X','e_2'):-3}))
        self.assertEqual(r2*3,self.T3.ext_alg.elt_from_cd({('X','e_2'):-3}))
        self.assertEqual(r2*0,r0)        
        
        self.assertEqual(4*s0,s0)
        self.assertEqual(0*s0,s0)
        self.assertEqual(0*s4,s0)
        self.assertEqual(-s1,self.T5.ext_alg.elt_from_cd({tuple():-3}))
        self.assertEqual(s1*(-1),self.T5.ext_alg.elt_from_cd({tuple():-3}))
        self.assertEqual(s0*4,s0)
        
        self.assertEqual(-c0,c0)
        self.assertEqual(-c1,self.C5.elt_from_cd({('Y',):-1}))
        self.assertEqual(5*c2,self.C5.elt_from_cd({('X','Y'):-15}))
        self.assertEqual(-2*c3,self.C5.elt_from_cd({('X','H'):-2,('N','e_3'):6}))
        self.assertEqual(c2*5,self.C5.elt_from_cd({('X','Y'):-15}))
        
        self.assertEqual(3*t0,t0)
        self.assertEqual(4*t1,self.T5.elt([4]+[0]*(len(self.T5.basis)-1)))
        self.assertEqual(4*b1,self.T5.elt([4]+[0]*(len(self.T5.basis)-1)))    
        self.assertEqual(t1*4,self.T5.elt([4]+[0]*(len(self.T5.basis)-1)))
        self.assertEqual(b1*4,self.T5.elt([4]+[0]*(len(self.T5.basis)-1)))
        self.assertEqual(t4*2,self.T5.elt([2,-4]+[0]*(len(self.T5.basis)-2)))
        
        # wedge
        self.assertEqual(s0.wedge(t0.cast_as_cochain()),c0)
        self.assertEqual(s4.wedge(t0.cast_as_cochain()),c0)
        self.assertEqual(s0.wedge(t3.cast_as_cochain()),c0)
        self.assertEqual(s1.wedge(t1.cast_as_cochain()),self.C5.elt_from_cd({('Y',):3}))
        self.assertEqual(s1.wedge(b1.cast_as_cochain()),self.C5.elt_from_cd({('Y',):3}))
        self.assertEqual(s2.wedge(t1.cast_as_cochain()),self.C5.elt_from_cd(
            {('e_2','Y'):2,('e_1','Y'):-1,('X','N','Y'):1}))
        
        self.assertEqual(r0.wedge(r1),r0)
        self.assertEqual(r1.wedge(r0),r0)
        self.assertEqual(r1.wedge(r2),r2)
        self.assertEqual(r2.wedge(r1),r2)
        self.assertEqual(r3.wedge(r3),r0)
        self.assertEqual(r2.wedge(r4),self.T3.ext_alg.elt_from_cd({('X','e_1','e_2'):1}))
        
        self.assertEqual(s0.wedge(s2),s0)
        self.assertEqual(s2.wedge(s0),s0)
        self.assertEqual(s1.wedge(s2),3*s2)
        self.assertEqual(s2.wedge(s1),3*s2)
        self.assertEqual(s2.wedge(s3),s4)
        
        self.assertEqual(s0.wedge(c1),c0)
        self.assertEqual(s0.wedge(c0),c0)
        self.assertEqual(s1.wedge(c0),c0)
        self.assertEqual(s1.wedge(c1),3*c1)
        self.assertEqual(s1.wedge(c2),3*c2)
        self.assertEqual(s2.wedge(c1),self.C5.elt_from_cd({
            ('e_2','Y'):2,('e_1','Y'):-1,('X','N','Y'):1}))
        self.assertEqual(s2.wedge(c2),self.C5.elt_from_cd({
            ('X','e_2','Y'):6,('X','e_1','Y'):-3}))
        
    def test_ker_proj(self):
        self.C3.init_basis(1)
        self.C3.init_basis(2)
        
        # The ker_proj should be in the kernel of coboundary
        # ker_proj shouldn't change the inner prod with ker elts
        for deg in [1,2]:
            for wght in self.C3.basis(deg):
                for c in self.C3.basis(deg,wght):
                    pc=self.C3.subspace_proj(c,'closed')
                    self.assertEqual(pc.cb(),self.C3.elt({}))
                    for k in range(sp.shape(self.C3.subspace_basis('closed',deg,wght))[1]):
                        t=self.C3.elt({deg:{wght:self.C3.subspace_basis('closed',deg,wght).col(k)}})
                        self.assertEqual(pc.iprod(t),c.iprod(t))
        # Also check for a couple of sums:
        c1=self.C3.elt_from_cd({('X','N','H'):1})
        c2=self.C3.elt_from_cd({('X','e_1','N'):1})
        c3=self.C3.elt_from_cd({('e_1','e_2','N'):1})
        
        d1=c1+8*c3
        d2=c1-2*c2+5*c3
        p1=self.C3.subspace_proj(d1,'closed')
        p2=self.C3.subspace_proj(d2,'closed')
        
        self.assertEqual(p1.cb(),self.C3.elt({}))
        self.assertEqual(p2.cb(),self.C3.elt({}))
        
        for deg in [1,2]:
            for wght in self.C3.basis(deg):
                for k in range(sp.shape(self.C3.subspace_basis('closed',deg,wght))[1]):
                    c=self.C3.elt({deg:{wght:self.C3.subspace_basis('closed',deg,wght).col(k)}})
                    self.assertEqual(p1.iprod(c),d1.iprod(c))
        
    def test_coker_proj(self):
        self.C3.init_basis(1)
        self.C3.init_basis(2)
        
        # coker_proj should be orthogonal to im(coboundary)
        # c-coker_proj(c) should be in im(coboundary),
        # which is the orthogonal complement of the cokernel
        for deg in [1,2]:
            for wght in self.C3.basis(deg):
                for c in self.C3.basis(deg,wght):
                    pc=self.C3.subspace_proj(c,'coclosed')
                    for k in range(sp.shape(self.C3.subspace_basis('exact',deg,wght))[1]):
                        a=self.C3.elt({deg:{wght:self.C3.subspace_basis('exact',deg,wght).col(k)}})
                        self.assertEqual(pc.iprod(a),0)
                        self.assertEqual((c-pc).iprod(pc),0)
                        
        # Also check for a couple of sums:
        c1=self.C3.elt_from_cd({('X','N','H'):1})
        c2=self.C3.elt_from_cd({('X','e_1','N'):1})
        c3=self.C3.elt_from_cd({('e_1','e_2','N'):1})
        
        d1=c1+8*c3
        d2=c1-2*c2+5*c3
        p1=self.C3.subspace_proj(d1,'coclosed')
        p2=self.C3.subspace_proj(d2,'coclosed')
        
        for deg in [1,2]:
            for wght in self.C3.basis(deg):
                for k in range(sp.shape(self.C3.subspace_basis('closed',deg,wght))[1]):
                    c=self.C3.elt({deg:{wght:self.C3.subspace_basis('closed',deg,wght).col(k)}})
                    self.assertEqual(p1.iprod(c),0)
                    self.assertEqual(p2.iprod(c),0)
                    self.assertEqual((d1-p1).iprod(p1),0)
                    self.assertEqual((d2-p2).iprod(p2),0)