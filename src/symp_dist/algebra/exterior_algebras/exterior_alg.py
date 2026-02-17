from __future__ import annotations
from typing import TYPE_CHECKING, Type

__all__ = ["ExtAlg"]

import sympy as sp
from ...utils import math_helpers as mh
from ..tensor_algebras import TensorAlg
from ...utils.exceptions import invalid_parent_exception
from itertools import combinations

if TYPE_CHECKING:
    from ..tanaka_symbols import TSymb
    from ..exterior_algebras import ExtElt


class ExtAlg(TensorAlg):
    def __init__(self, T_symb_obj:TSymb):
        from ..exterior_algebras.exterior_elt import ExtElt

        T_symb_obj.ext_alg = self
        TensorAlg.__init__(self, T_symb_obj)

    @property
    def childcls(self) -> Type[ExtElt]:
        from ..exterior_algebras import ExtElt
        return ExtElt 


    def wedge_tuples(self, tuple1:tuple, tuple2:tuple, obj2_type:str)-> tuple[tuple[str,...],int] | str:
        """Returns (wedge,sgn), where wedge is a tuple representing
        tuple1 wedge tuple2 and sgn is -1 or 1

        INPUTS:
        * 'tuple1', 'tuple2' - tuples of TSymbBasisElt objects
        * 'obj2_type' - among 'ext_elt' and 'cochain', indicating the type of
                        the object repped by tuple2"""
        if obj2_type not in ["ext_elt", "cochain"]:
            raise invalid_parent_exception(
                "wedge_tuples recieved invalid parent type as arg"
            )
        if obj2_type == "ext_elt":
            B = tuple2
        else:
            B = tuple(list(tuple2)[0:-1])
        # check for repeats
        if len(set(tuple1).union(set(B))) != len(tuple1) + len(B):
            return "Nil"
        t, s = mh.sort_basis_tuple(tuple1 + B, self.alg.basis_strs)
        if obj2_type == "ext_elt":
            return (t, s)
        return (t + (tuple2[-1],), s)

    def tuple_wght(self, t:tuple)->int:
        """Returns the weight of t
        INPUTS:
        * 't' - a tuple representing an element of self
        """
        r = 0
        for A in t:
            i = self.alg.basis_strs.index(A)
            r = r - self.alg.wght_list[i]  # This is the exterior algebra of m_dual
        return r

    def tuple_deg(self, t:tuple)->int:
        """Returns the degree of t
        INPUTS:
        * 't' - a tuple representing an element of self
        """
        return len(t)

    def dwi(self, basis_str_tuple:tuple[str,...])->tuple[int,int,int]:
        """Returns the degree, weight, and index of basis_tuple
        Inputs:
        * 'basis_tuple' -- a tuple of basis strings representing a basic cochain
        """
        if basis_str_tuple not in self.dwi_dicts:
            self.init_basis(len(basis_str_tuple))
        return self.dwi_dicts[basis_str_tuple]

    def sort_tuple(self, t:tuple[str,...])->tuple:
        """Returns an (ext alg) sorting of t and the sign of the corresponding permutation, as a tuple
        or None if t contains repeats
        INPUTS:
        * 't' - a tuple of basis_strs
        """
        if len(t) != len(set(t)):
            return (None, 0)
        return mh.sort_basis_tuple(t, self.alg.basis_strs)

    def init_basis(self, deg:int)->None:
        from ..exterior_algebras.exterior_alg_basis_elt import ExtBasisElt

        """Sets value of deg in basis_cache and adds to basis_dicts"""
        if deg in self.basis_cache:
            return None
        deg_subsets = [
            A for A in list(combinations([str(A) for A in self.alg.m_basis], deg))
        ]
        self.basis_cache[deg] = {}
        wght_ct = {}
        for i in range(len(deg_subsets)):
            # count the number of elements of deg d and wght w
            # and set the deg, wght, and index of each elt in dwi_dicts
            A = deg_subsets[i]
            w = self.tuple_wght(A)
            if w not in wght_ct:
                j = 0
                wght_ct[w] = 1
            else:
                j = wght_ct[w]
                wght_ct[w] += 1
            self.dwi_dicts[A] = (deg, w, j)
        for A in deg_subsets:
            d, w, i = self.dwi_dicts[A]
            vec = sp.SparseMatrix(sp.zeros(wght_ct[w], 1))
            vec[i] = 1
            vd = {d: {w: vec}}
            b_elt = ExtBasisElt(self, d, w, vd, A)
            if i == 0:
                self.basis_cache[deg][w] = [b_elt]
            else:
                self.basis_cache[deg][w].append(b_elt)

    def wedge_indices(self, d1:int, w1:int, i1:int,
                       d2:int, w2:int, i2:int, obj2_type:str)->tuple:
        """Returns a tuple with deg, wght, index, and sign of the tuples
        or ('Nil','Nil','Nil',0) if the wedge is zero
        INPUTS:
        * 'd1','d2' - degrees
        * 'w1','w2' - weights
        * 'i1','i2' - indices
        * 'obj2_type' - either 'ext_elt' or 'cochain', giving the type of
                        the object specified by (d2,w2,i2)
        """
        if obj2_type not in ["ext_elt", "cochain"]:
            raise invalid_parent_exception(
                "wedge_indices recieved invalid parent as arg"
            )

        t1 = tuple([str(A) for A in self.basis(d1, w1)[i1].components])
        if obj2_type == "ext_elt":
            t2 = tuple([str(A) for A in self.basis(d2, w2)[i2].components])
        else:
            t2 = tuple(
                [str(A) for A in self.alg.cochain_complex.basis(d2, w2)[i2].components]
            )

        r = self.wedge_tuples(t1, t2, obj2_type)
        if r == "Nil" or isinstance(r,str): # I've just included this for type checking resolution
            return ("Nil", "Nil", "Nil", 0)
        t3, s = (r[0], r[1])  # resulting tuple and sign
        if obj2_type == "ext_elt":
            i3 = self.basis_strs(d1 + d2, w1 + w2).index(t3)
        else:
            i3 = self.alg.cochain_complex.basis_strs(d1 + d2, w1 + w2).index(t3)
        return (d1 + d2, w1 + w2, i3, s)
