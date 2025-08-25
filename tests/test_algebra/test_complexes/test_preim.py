import unittest
import random
from src.algebra.tanaka_symbols.symplectic_symbol import SympSymb

class TestPreim(unittest.TestCase):
    def test_preim(self):
        self.T5=SympSymb(5)
        self.C5=self.T5.cochain_complex
        
        # Construct a list of random cochains
        c_list=[self.C5.elt({})]
        for i in range(10):
            new_c=self.C5.elt({})
            for deg in [1,2,3]:
                for wght in self.C5.basis(deg):
                    if random.randint(0,2)==2:
                        for A in self.C5.basis(deg,wght):
                            if random.randint(0,1)==1:
                                new_c+=random.randint(-10,10)*A
            c_list.append(new_c)
            
        for c in c_list:
            c_proj=self.C5.subspace_proj(c,'exact')
            self.assertEqual((self.C5.cb_preim_elt(c_proj)).cb(),c_proj)