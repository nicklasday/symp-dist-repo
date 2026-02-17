# Commented out for time purposes
import unittest
import sympy as sp
from symp_dist.utils import lin_alg_helpers as lh
from symp_dist.algebra.tanaka_symbols.symplectic_symbol import SympSymb

class TestSubspaces(unittest.TestCase):
    def test_subspaces_orthogonality(self):
        self.T5=SympSymb(5)
        self.C5=self.T5.cochain_complex
        
        # coker and im should be perpindicular
        for deg in [1,2]:
            for wght in self.C5.basis(deg):
                for k in range(sp.shape(self.C5.subspace_basis('coclosed',deg,wght))[1]):
                    elt=self.C5.elt({deg:{wght:self.C5.subspace_basis('coclosed',deg,wght).col(k)}})
                    self.assertEqual(self.C5.subspace_proj(elt,'exact'),self.C5.elt({}))
        
        for deg in [1,2]:
            for wght in self.C5.basis(deg):
                for k in range(sp.shape(self.C5.subspace_basis('exact',deg,wght))[1]):
                    elt=self.C5.elt({deg:{wght:self.C5.subspace_basis('exact',deg,wght).col(k)}})
                    self.assertEqual(self.C5.subspace_proj(elt,'coclosed'),self.C5.elt({}))
        
        # harmonic forms should be be perpindicular to im and coim
        for deg in [1,2]:
            for wght in self.C5.basis(deg):
                for k in range(sp.shape(self.C5.subspace_basis('harmonic',deg,wght))[1]):
                    elt=self.C5.elt({deg:{wght:self.C5.subspace_basis('harmonic',deg,wght).col(k)}})
                    self.assertEqual(self.C5.subspace_proj(elt,'exact'),self.C5.elt({}))
                    self.assertEqual(self.C5.subspace_proj(elt,'coexact'),self.C5.elt({}))

## Whoops, I seem to have written two of these?

class test_subspaces(unittest.TestCase):
    # This shows that the subspaces have the correct orthogonality and containment relations
    # These tests are passable for subspaces smaller than the desired ones 
    def setUp(self):
        self.T=SympSymb(5)
        self.C=self.T.cochain_complex
    
    def test_closed(self):
        # zero coboundary
        for d in [1,2]:
            for w in self.C.basis(d):
                closed=self.C.subspace_basis('closed',d,w)
                for i in range(sp.shape(closed)[1]):
                    self.assertEqual(self.C.elt({d:{w:closed.col(i)}}).cb(),self.C.elt({}))
        
    def test_exact(self):
        # To Do
        # contained in closed
        # contained in image of coboundary
        for d in [1,2]:
            for w in self.C.basis(d):
                closed=self.C.subspace_basis('closed',d,w)
                exact=self.C.subspace_basis('exact',d,w)
                self.assertTrue(lh.colspace_containment(exact,closed))
                
                im=sp.Matrix([list(c.cb_vec()) for c in self.C.basis(d-1,w)]).transpose()
                self.assertTrue(lh.colspace_containment(exact,im))
    
    def test_coclosed(self):
        # orthogonal to exact
        for d in [1,2]:
            for w in self.C.basis(d):
                exact=self.C.subspace_basis('exact',d,w)
                coclosed=self.C.subspace_basis('coclosed',d,w)
                A=exact.transpose()*self.C.Q(d,w)*coclosed
                self.assertEqual(A,sp.zeros(*sp.shape(A)))
        
    def test_coexact(self):
        for d in [1,2]:
            for w in self.C.basis(d):
                # orthogonal to closed
                coexact=self.C.subspace_basis('coexact',d,w)
                closed=self.C.subspace_basis('closed',d,w)
                A=coexact.transpose()*self.C.Q(d,w)*closed
                self.assertEqual(A,sp.zeros(*sp.shape(A)))
                
                # Contained in coclosed
                coclosed=self.C.subspace_basis('coclosed',d,w)
                self.assertTrue(lh.colspace_containment(coexact,coclosed))