import unittest
import sympy as sp
from symp_dist.algebra.tanaka_symbols.symplectic_symbol import SympSymb
from symp_dist.distributions.particular_distributions import Free_distr
from symp_dist.cartan_geometries.geom_prolongation import Geom_Prolongation

class TestApplyCochainMap(unittest.TestCase):
    def setUp(self):
        self.K=sp.IndexedBase('K')
        self.T=SympSymb(5)
        self.C=self.T.cochain_complex
        self.D=Free_distr(self.T)[0]
        self.P=Geom_Prolongation(self.D,{})
        
    def test_apply_cochain_map_base(self):
        for a in self.T.basis[3:len(self.T.basis)]:
            for b in self.T.basis:
                for c in self.T.basis:
                    if a.wght<0 and b.wght<0 and a!=b:
                        coch0=self.C.elt_from_cd({(str(a),str(b),str(c)):1})
                        coch1=self.C.elt_from_cd({(str(a),str(b)):1})
                        coch2=self.C.elt_from_cd({(str(a),str(b),'Y'):1})

                        self.assertEqual(coch0.apply_cochain_map_base(a,b),c)
                        self.assertEqual(2*coch0.apply_cochain_map_base(a,b),2*c)
                        self.assertEqual(coch0.apply_cochain_map_base(a),0)
                        if c!=a and c.wght<0: self.assertEqual(coch0.apply_cochain_map_base(c,b),0)
                        self.assertEqual((coch0+2*coch2).apply_cochain_map_base(a,b),c+2*self.T.basis[0])
                        self.assertEqual((2*coch0-coch2).apply_cochain_map_base(a,b),2*c-self.T.basis[0])
                        self.assertEqual(coch1.apply_cochain_map_base(a),b)
                        self.assertEqual((2*coch1+coch0).apply_cochain_map_base(a),2*b)
                        self.assertEqual((2*coch1+coch0).apply_cochain_map_base(a,b),c)


    def test_apply_cochain_map_linearize(self):
        # Making sure things work on the exterior algebra, and a few linear combinations in the arguments
        for a in self.T.basis[3:len(self.T.basis)]:
            for b in self.T.basis[3:len(self.T.basis)]:
                for c in self.T.basis[3:len(self.T.basis)]:
                    if a.wght<0 and b.wght<0 and a!=b:
                        coch0=self.C.elt_from_cd({(str(a),str(b),str(c)):1})
                        coch1=self.C.elt_from_cd({(str(a),str(b)):1})
                        coch2=self.C.elt_from_cd({(str(a),str(b),'Y'):1})
                        awb=a.cast_as_ext_elt().wedge(b.cast_as_ext_elt())
                        awc=a.cast_as_ext_elt().wedge(c.cast_as_ext_elt())
                        bwc=b.cast_as_ext_elt().wedge(c.cast_as_ext_elt())

                        self.assertEqual(coch0.apply_cochain_map(awb),c)
                        self.assertEqual(coch1.apply_cochain_map(awb),self.T.elt())
                        self.assertEqual(coch2.apply_cochain_map(awb),self.T.basis[0])
                        if c==b:self.assertEqual(coch0.apply_cochain_map(awc),c)
                        else: self.assertEqual(coch0.apply_cochain_map(awc),self.T.elt())
                        self.assertEqual(coch1.apply_cochain_map(awc),self.T.elt())
                        if c==a: self.assertEqual(coch0.apply_cochain_map(bwc),-c)
                        else: self.assertEqual(coch0.apply_cochain_map(bwc),self.T.elt())
                        if c!=b:
                            self.assertEqual(coch0.apply_cochain_map(-2*awb+3*awc+4*a.cast_as_ext_elt()),-2*c)


                        self.assertEqual(coch1.apply_cochain_map(a.cast_as_ext_elt()),b)
                        if b!=a: self.assertEqual(coch1.apply_cochain_map(3*a.cast_as_ext_elt()-b.cast_as_ext_elt()+awb),3*b)
