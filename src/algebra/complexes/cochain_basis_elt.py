from __future__ import annotations
from typing import TYPE_CHECKING
__all__ = ["CochainBasisElt"]

import sympy as sp

from ...utils import math_helpers as mh
from .cochain import Cochain
from ..tensor_algebras.tensor_alg_basis_elt import TensorAlgBasisElt
from itertools import combinations

if TYPE_CHECKING:
    from .cochain_complex import CochainComplex
    from ..exterior_algebras.exterior_alg_basis_elt import ExtBasisElt
    from ...algebra.tanaka_symbols.tanaka_symbol_elt import TSymbElt



class CochainBasisElt(Cochain, TensorAlgBasisElt):
    def __init__(self, parent:CochainComplex, deg:int, wght:int, vd:dict[int,dict[int,sp.MatrixBase]], str_rep:tuple[str,...]):
        """Initializes a new CochainBasisElt instance.
        
        Args:
            parent (CochainComplex): A complex to which the element will belong
            deg (int): The degree of the element (for example, )"""
        
        Cochain.__init__(self, parent, vd)
        TensorAlgBasisElt.__init__(self, parent, deg, wght, vd, str_rep)
        self.length:int = -1
        self.parent:CochainComplex
        if parent.alg.Q is not None:
            self.length = sp.Rational(
                self.components[-1].length,
                sp.prod([A.length for A in self.components[0:-1]]),
            )

        # Set the symbol representation of self
        sl = [A.symb_str_rep for A in self.components]
        s = "{" + sl[0] + r"^*}"
        for i in range(1, len(sl) - 1):
            s += r"\wedge{" + sl[i] + r"^*}"
        s += r"\otimes{" + sl[-1] + "}"
        self.symb:sp.Symbol = sp.symbols(s)

    def cb_vec(self)->sp.MatrixBase:
        """Returns the coboundary of self; should only be called by cochain.cb_mat.

        Returns:
            sp.Matrix: A column matrix representing the coboundary of self in the basis of appropriate weight and degree
        """
        mb = self.parent.alg.m_basis
        Y = self.components[-1]
        r = [0] * len(self.parent.basis(self.deg + 1, self.wght))

        B_set = set([mb.index(A) for A in self.components[0:-1]])
        non_B_set = set(range(len(mb))).difference(B_set)

        # First term
        for a in non_B_set:
            Xa = mb[a]
            # compute the RHS above, named im_a_vec: [X_a,Y]
            im_a_vec = Xa.ad(Y).vec
            for j in range(len(im_a_vec)):
                if im_a_vec[j] != 0:
                    tl = [Xa] + self.components[0:-1] + [self.parent.alg.basis[j]]
                    t, s = self.parent.sort_tuple(tuple([str(A) for A in tl]))
                    if s != 0:
                        ind = self.parent.basis_strs(self.deg + 1, self.wght).index(t)
                        r[ind] += s * im_a_vec[j]
        # Second term
        for a in non_B_set:
            for b in non_B_set:
                if a < b:
                    Xa = mb[a]
                    Xab = Xa.ad_mat(mod="m").col(b)
                    for i in B_set:
                        if Xab[i] != 0:
                            cpts = [str(A) for A in self.components[0:-1]]
                            cpts.remove(str(mb[i]))
                            tl = (
                                [str(Xa), str(mb[b])]
                                + cpts
                                + [str(self.components[-1])]
                            )
                            t, s = self.parent.sort_tuple(tuple(tl))
                            if s != 0:
                                ind = self.parent.basis_strs(
                                    self.deg + 1, self.wght
                                ).index(t)
                                r[ind] += (
                                    (-1) ** (1 + self.components.index(mb[i]))
                                    * s
                                    * Xab[i]
                                )

        return sp.SparseMatrix([r]).transpose()


def Apply_Gerst_prod_on_base(f:Cochain, h:Cochain, w:ExtBasisElt)->TSymbElt:
    """returns the Gerstenhaber product of f and h applied to w

    Args:
        f (Cochain): a homogeneous cochain
        h (Cochain): a homogeneous cochain
        w (ExtBasisElt): a basis element in the exterior algebra
        
    Returns:
        TSymbElt: the Gerstenhaber product of f and h applied to w"""
    
    if f == f.parent.elt({}) or h == h.parent.elt({}) or w == w.parent.elt({}):
        return f.parent.alg.elt()
    fd = list(f.vd.keys())[0]
    hd = list(h.vd.keys())[0]

    if w.deg != fd + hd - 1:
        return w.parent.alg.elt([0] * len(w.parent.alg.basis))
    r = 0
    for comb in list(combinations(w.components, hd)):
        l1 = list(comb)
        l2 = [a for a in w.components if a not in comb]
        s = mh.permutation_sign(l1 + l2, w.components)

        if len(l1) == 0:
            w1 = f.parent.alg.ext_alg.basis(0, 0)[0]
        else:
            w1 = l1[0].cast_as_ext_elt()
        if len(l2) == 0:
            w2 = f.parent.alg.ext_alg.basis(0, 0)[0]
        else:
            w2 = l2[0].cast_as_ext_elt()

        for t in l1[1 : len(l1)]:
            w1 = w1.wedge(t.cast_as_ext_elt())
        for t in l2[1 : len(l2)]:
            w2 = w2.wedge(t.cast_as_ext_elt())
        t = f.apply_cochain_map(
            h.apply_cochain_map(w1).negative_projection().cast_as_ext_elt().wedge(w2)
        )
        r = r + s * t
    return r
