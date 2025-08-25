__all__ = ["RegularCartanGeometry"]

import sympy as sp
from ..distributions.diff_op_dict import DiffOpDict
from ..utils import ds_helpers as dsh
from ..algebra.tanaka_symbols import TSymbElt,TSymb


class RegularCartanGeometry(object):
    def __init__(self, g:TSymb, curv_str:str="K", coords=None):
        """Constructs a CartanGeometry of with the given curvature and symbol

        Arguments:
            * 'g' - a graded Lie algebra
            * 'c' - a 2-cochain from the complex C_+(g,g)

        Only intended for use in the computation of the Bianchi identity
        """
        self.symbol = g
        if coords is None:
            p = len(g.basis)
            self.coords = sp.symbols("x_0:{}".format(p))
        #self.ndo_dict = {}
        self.curvature = g.cochain_complex.regular_normal_2_cochain(curv_str)
        self.Bianchi_cache:dict = {} # This seems to be unused? To do: Make use of this!

        self.fund_der_cache:list[DiffOpDict] = []
        self.fund_invars:sp.Indexed = [] # Apparently unused
        #self.der_cache = {} # Apparently unused

    def queryJacobi(self)->bool:
        """Returns True if the Jacobi identity holds for the fundamental vector fields"""
        raise NotImplementedError

    def fund_der(self, f, i):
        """Returns the fundamental derivative in direction i of the function f

        Arguments:
            * 'f' - a rational function in terms of 2-tensors
            * 'ind' - an integer index for a direction
        """
        if type(f) == TSymbElt:
            return f.parent.elt([self.fund_der(fj, i) for fj in f.vec])
        self.init_fund_der(i)
        return self.fund_der_cache[i].apply(f, normal=True)

    def init_fund_der(self, i):
        """Caches fundamental derivative DiffOpDicts up to index i"""
        if len(self.fund_der_cache) > i:
            return None
        self.compute_fund_der(i)
        return None

    def compute_fund_der(self, i):
        for j in range(len(self.fund_der_cache), i + 1):
            self.fund_der_cache.append(
                DiffOpDict({(j,): 1}, self.symbol, self.curvature).normal_form()
            )
        return None

    def update_fund_ders(self, ds_dict, distr):
        for k in range(len(self.fund_der_cache)):
            for t in self.fund_der_cache[k].d:
                self.fund_der_cache[k].d[t] = dsh.ds_subs(
                    self.fund_der_cache[k].d[t], ds_dict
                )[0]
        return None


    def der_term(self, i0:int, i1:int, i2:int):
        """Returns the derivative term for the component of the Bianchi identity
        applied to (i0,i1,i2), which is an element of P.symbol."""
        g:TSymb = self.symbol
        X0, X1, X2 = [g.basis[a] for a in [i0, i1, i2]]
        w0, w1, w2 = [None, None, None]
        try:
            w0 = g.ext_alg.elt_from_cd({(str(X1), str(X2)): 1})
        except KeyError:
            pass
        try:
            w1 = g.ext_alg.elt_from_cd({(str(X2), str(X0)): 1})
        except KeyError:
            pass
        try:
            w2 = g.ext_alg.elt_from_cd({(str(X0), str(X1)): 1})
        except KeyError:
            pass

        r = g.elt()
        if w0 is not None:
            r += self.fund_der(self.curvature.apply_cochain_map(w0), i0)
        if w1 is not None:
            r += self.fund_der(self.curvature.apply_cochain_map(w1), i1)
        if w2 is not None:
            r += self.fund_der(self.curvature.apply_cochain_map(w2), i2)
        return r


    def cb_term(self, i0, i1, i2):
        """returns the coboundary of P.curvature applied to i0,i1,i2 elements of P.symbol.basis.
        This is needed because we care about the value as a cochain in C(g,g), not just C(m,g)
        (at least for the purpose of checks)"""

        r = self.symbol.elt()
        X0, X1, X2 = [self.symbol.basis[a] for a in [i0, i1, i2]]

        for i in range(3):
            Y0, Y1, Y2 = [X0, X1, X2, X0, X1][i : i + 3]
            w0 = (
                Y0.negative_projection()
                .cast_as_ext_elt()
                .wedge(Y1.negative_projection().cast_as_ext_elt())
            )
            r += self.curvature.apply_cochain_map(w0).ad(Y2)
            w1 = (
                Y0.ad(Y1)
                .negative_projection()
                .cast_as_ext_elt()
                .wedge(Y2.negative_projection().cast_as_ext_elt())
            )
            r += self.curvature.apply_cochain_map(w1)
        return -r


    # What if instead of computing these individually, I just computed Gerstenhaber square
    # of curvature once, then
    def gerst_term(self, i0, i1, i2):
        X0, X1, X2 = [self.symbol.basis[a] for a in [i0, i1, i2]]
        r = self.symbol.elt()
        for i in range(3):
            Y0, Y1, Y2 = [X0, X1, X2, X0, X1][i : i + 3]
            Y0 = Y0.negative_projection().cast_as_ext_elt()
            Y1 = Y1.negative_projection().cast_as_ext_elt()
            Y2 = Y2.negative_projection().cast_as_ext_elt()
            w = (
                self.curvature.apply_cochain_map(Y0.wedge(Y1))
                .negative_projection()
                .cast_as_ext_elt()
            )
            r += self.curvature.apply_cochain_map(w.wedge(Y2))
        return r


    def Bianchi(self, i0, i1, i2, subdivide=False):
        r0 = self.cb_term(i0, i1, i2)
        r1 = self.der_term(i0, i1, i2)
        r2 = self.gerst_term(i0, i1, i2)
        if subdivide:
            return (r0, r1, r2)
        return r0 + r1 + r2
        # if subdivide: return (cb_term(P,i0,i1,i2), der_term(P,i0,i1,i2), gerst_term(P,i0,i1,i2))
        # return cb_term(P,i0,i1,i2)+der_term(P,i0,i1,i2)+gerst_term(P,i0,i1,i2)
