from __future__ import annotations
from typing import TYPE_CHECKING, Type

__all__ = ["TSymb"]

from ...utils.sympy_utils import to_sympy_matrix
import sympy as sp
from ...utils.exceptions import invalid_parent_exception

if TYPE_CHECKING:
    from ...utils.types import VectorInput
    from tanaka_symbol_elt import TSymbElt
    from ..tensor_algebras.tensor_alg_elt import TensorAlgElt


class TSymb(object):
    def __init__(self, basis_strs, symb_strs, wght_list, ad_matrices, Q=None):
        from ..complexes.cochain_complex import CochainComplex
        from ..exterior_algebras.exterior_alg import ExtAlg
        from .tanaka_symbol_basis_elt import TSymbBasisElt

        self.basis_strs:list[str] = basis_strs
        self.wght_list:list[int] = wght_list
        self.Q:sp.MatrixBase = Q
        self.basis:list[TSymbBasisElt] = [
            TSymbBasisElt(
                self.basis_strs[i],
                symb_strs[i],
                self.wght_list[i],
                self,
                i,
                ad_matrices[i],
            )
            for i in range(len(self.basis_strs))
        ]
        self.ext_alg:ExtAlg = ExtAlg(self)
        self.cochain_complex:CochainComplex = CochainComplex(self)
        self.m_basis:list[TSymbBasisElt] = [
            self.basis[i] for i in range(len(self.basis)) if self.wght_list[i] < 0
        ]
        self.m_basis_strs:list[str] = [str(A) for A in self.m_basis]

    @property
    def childcls(self) -> Type[TensorAlgElt]:
        return TSymbElt 

    def jacobi_test(self):
        """returns: True if the Jacobi identity holds, False otherwise"""
        for A in self.basis:
            for B in self.basis:
                for C in self.basis:
                    t1 = self.ad(A, self.ad(B, C))
                    t2 = self.ad(self.ad(A, B), C) + self.ad(B, self.ad(A, C))
                    if t1 != t2:
                        return False
        return True

    def elt(self, vec:VectorInput=None):
        from .tanaka_symbol_elt import TSymbElt

        if vec is None:
            return TSymbElt([0] * len(self.basis), self)
        return TSymbElt(to_sympy_matrix(vec), self)

    def mod_dim(self, mod="g", deg=None, wght=None):
        """Returns the dimension of the specified module, with degree and wght possibly specified
        INPUTS:
        * 'mod' - the desired module, among 'g','m','g_dual','m_dual','E','CE'
        * 'deg' - degree
        * 'wght' - wght
        """
        if deg is None and wght is None:
            if mod in ["g", "g_dual"]:
                return len(self.basis)
            if mod in ["m", "m_dual"]:
                return len(self.m_basis)
            if deg is None:
                print("degree must be specified to compute mod_dim of E or CE")
            if mod == "E":
                return sp.binomial(len(self.m_basis), deg)
            if mod == "CE":
                return sp.binomial(len(self.m_basis), deg) * len(self.basis)

        if deg is None and wght is not None:
            if mod == "g":
                return len([A for A in self.basis if A.wght == wght])
            if mod == "g_dual":
                return len([A for A in self.basis if -A.wght == wght])
            if mod == "m":
                return len([A for A in self.m_basis if A.wght == wght])
            if mod == "m_dual":
                return len([A for A in self.m_basis if -A.wght == wght])
            if mod in ["E", "CE"]:
                raise NotImplementedError

        if mod in ["g", "g_dual", "m", "m_dual"]:
            print("do not specify deg when computing dim(m) or dim(g)")

        if wght is None:
            if mod == "E":
                return sp.binomial(len(self.m_basis), deg)
            if mod == "CE":
                return sp.binomial(len(self.m_basis), deg) * len(self.basis)

        if mod == "E":
            return len(self.ext_alg.basis(deg, wght))
        if mod == "CE":
            return len(self.cochain_complex.basis(deg, wght))

        print("mod_dim module must be among g, m, g_dual, m_dual, E, and CE")

    def iprod(self, t1, t2):
        """t1,t2: of the same type among TSymbBasisElt, TSymbElt, ext_elt, and cochain"""
        if not hasattr(t1, "parent") and hasattr(t2, "parent"):
            raise ValueError(
                "arguments of iprod must have type TSymbBasisElt, TSymbElt, ext_elt, or cochain"
            )
        if self.Q is None:
            print("Inner product matrix Q not initialized for", self)
        if t1 == 0 or t2 == 0:
            return 0
        if t1.parent == t2.parent:
            return (t1.vec.transpose() * self.Q * t2.vec)[0]
        raise invalid_parent_exception("arguments of iprod must have the same parent")

    @staticmethod
    def ad(A1, A2):
        return A1.ad(A2)