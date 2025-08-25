from __future__ import annotations
from typing import TYPE_CHECKING, Type, TypeVar, Generic, overload, Optional, Union

__all__ = ["TensorAlg"]

import sympy as sp
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .tensor_alg_elt import TensorAlgElt
    from ..tanaka_symbols import TSymb

T = TypeVar('T', bound='TensorAlgElt')

class TensorAlg(ABC,Generic[T]):
    def __init__(self, T_symb_obj):
        self.alg:TSymb = T_symb_obj
        self.basis_cache = {}
        self.dwi_dicts = {}  # Keys: str_reps of basis elts; Values:(deg,wght,index)
        # self.childcls: Type[TensorAlgElt] = None

    @property
    @abstractmethod
    def childcls(self) -> Type[T]:
        """Subclasses must define this as a class that inherits from TensorAlgElt"""
        pass

    @abstractmethod
    def init_basis(self, deg:int)->None:
        """Initializes a basis for this algebra in fixed degree"""
        pass

    def elt(self, vd:dict[int,dict[int,sp.Expr]])->T:
        return self.childcls(self, vd)

    def elt_from_cd(self, cd:dict[tuple[str,...],sp.Expr]={})->T:
        return self.childcls.from_cd(self, cd)

    def basis(self, deg:int, wght:int|None=None):
        """Returns a basis for (deg, wght) as a list, or if wght==None, returns
        basis for deg as a dict of wghts."""
        if deg < 0:
            return []
        self.init_basis(deg)
        if wght is None:
            return self.basis_cache[deg]
        if deg not in self.basis_cache or wght not in self.basis_cache[deg]:
            return []
        return self.basis_cache[deg][wght]

    @overload
    def basis_strs(self,deg:int)->dict[int,list[tuple[str,...]]]:...

    @overload
    def basis_strs(self, deg:int, wght:int)->list[tuple[str,...]]:...

    def basis_strs(self, deg: int, wght: Optional[int] = None) -> Union[
        dict[int, list[tuple[str, ...]]], list[tuple[str, ...]]
    ]:
        b = self.basis(deg, wght)
        if wght is None:
            r = {}
            for k in b:
                r[k] = [tuple([str(c) for c in A.components]) for A in b[k]]
            return r
        return [tuple([str(c) for c in A.components]) for A in b]

    def tuple_deg(self, t):
        NotImplemented

    def tuple_wght(self, t):
        NotImplemented

    def sort_tuple(self, t):
        NotImplemented

    def cd_to_vd(self, cd:dict[tuple[str,...],sp.Expr]={})->dict[int,dict[int,sp.MatrixBase]]:
        """Converts a coeff dict to a vector dict in the basis of self"""
        r:dict[int,dict[int,sp.MatrixBase]] = {}
        for A in cd:
            d = self.tuple_deg(A)
            w = self.tuple_wght(A)
            A1, s = self.sort_tuple(A)
            if d not in r:
                r[d] = {}
            if w not in r[d]:
                r[d][w] = sp.SparseMatrix(sp.zeros(len(self.basis(d, w)), 1))
            v = sp.SparseMatrix(sp.zeros(len(self.basis(d, w)), 1))
            v[self.dwi_dicts[A1][2]] = s * cd[A]
            r[d][w] = r[d][w] + v
        return r

    def Q(self, d:int, w:int)->sp.MatrixBase:
        """Returns the inner product matrix induces by that on the
        base algebra for degree d and weight w
        INPUTS:
        * 'd' - a degree
        * 'w' - a weight
        """
        if self.alg.Q is None:
            print("Inner product on base algebra not initialized")
        return sp.SparseMatrix(sp.diag(*[A.length for A in self.basis(d, w)]))

    def iprod(self, elt1:T, elt2:T)->sp.Expr:
        """Returns the inner product of elt1 and elt2
        INPUTS:
        * 'elt1', 'elt2' - elements of self
        """
        return elt1.iprod(elt2)
