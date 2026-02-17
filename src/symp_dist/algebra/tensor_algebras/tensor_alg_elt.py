from __future__ import annotations
from typing import TYPE_CHECKING

__all__ = ["TensorAlgElt"]

import sympy as sp
from ...utils import lin_alg_helpers as lah
import copy
from ...utils.exceptions import invalid_parent_exception
from ...utils.sympy_utils import to_sympy_matrix

if TYPE_CHECKING:
    from ...utils.types import VectorInput
    from typing import Self

class TensorAlgElt(object):
    from .tensor_alg import TensorAlg
    def __init__(self, parent:TensorAlg, vd:dict[int,dict[int,VectorInput]]={}):
        """INPUTS:
        * 'parent' - A tensor algebra
        * 'vd' - a dict of dicts of vectors, keyed by degree then weight"""
        self.parent = parent
        self.vd = vd
        for i in self.vd:
            for j in self.vd[i]:
                self.vd[i][j]=to_sympy_matrix(self.vd[i][j])


    @classmethod
    def from_cd(cls, parent, cd={}):
        NotImplemented

    def __str__(self)->str:
        return lah.str_from_vd(self.vd, self.parent.basis)

    def __repr__(self)->str:
        return self.__str__()

    def __eq__(self, other)->bool:
        if type(other) == type(None):
            return False
        z = self - other
        return z.is_zero()

    def is_zero(self)->bool:
        for d in self.vd:
            for w in self.vd[d]:
                if sp.simplify(self.vd[d][w]) != sp.zeros(*sp.shape(self.vd[d][w])):
                    return False
        return True

    def __neg__(self)->Self:
        nvd = {d: copy.copy(self.vd[d]) for d in self.vd}
        for d in nvd:
            for w in nvd[d]:
                nvd[d][w] = -nvd[d][w]
        return self.parent.elt(nvd)

    def __add__(self, other)->Self:
        nvd:dict[int,dict[int,sp.Matrix]] = {}
        for d in set(self.vd.keys()).union(other.vd.keys()):
            nvd[d] = {}
            if d in self.vd:
                if d in other.vd:
                    for w in set(self.vd[d].keys()).union(set(other.vd[d].keys())):
                        if w in self.vd[d]:
                            if d not in nvd:
                                nvd[d] = {}
                            nvd[d][w] = copy.copy(self.vd[d][w])
                            if w in other.vd[d]:
                                nvd[d][w] = nvd[d][w] + other.vd[d][w]
                        else:
                            nvd[d][w] = other.vd[d][w]
                else:
                    nvd[d] = copy.copy(self.vd[d])
            else:
                nvd[d] = copy.copy(other.vd[d])
        return self.parent.elt(nvd)

    def __radd__(self, other)->Self:
        return self + other

    def __sub__(self, other)->Self:
        return self + (-other)

    def __mul__(self, k)->Self:
        nvd:dict[int,dict[int,sp.Matrix]] = {}
        for d in self.vd:
            nvd[d] = {}
            for w in self.vd[d]:
                try:
                    nvd[d][w] = k * self.vd[d][w]
                except:
                    nvd[d][w] = sp.zeros(*sp.shape(self.vd[d][w]))
                    for j in range(len(nvd[d][w])):
                        nvd[d][w][j] = k * self.vd[d][w][j]
        return self.parent.elt(nvd)

    def __rmul__(self, other)->Self:
        return self * other

    def clear_zeros(self)->None:
        """clears zeros from the vector dict of self, leaving only necessary keys"""

        for d in self.vd:
            zero_keys = []
            for w in self.vd[d]:
                if self.vd[d][w] == sp.zeros(*self.vd[d][w].shape):
                    zero_keys.append(w)
            for w in zero_keys:
                self.vd[d].pop(w)

        zero_keys = []
        for d in self.vd:
            if self.vd[d] == {}:
                zero_keys.append(d)
        for d in zero_keys:
            self.vd.pop(d)

    def iprod(self, other)->sp.Expr:
        if not hasattr(other, "parent"):
            raise ValueError("arg of iprod must have parent")
        if self.parent != other.parent:
            raise invalid_parent_exception("args of iprod must have the same parent")
        r = 0
        for d in set(self.vd.keys()).intersection(set(other.vd.keys())):
            for w in set(self.vd[d].keys()).intersection(set(other.vd[d].keys())):
                r += (self.vd[d][w].transpose() * self.parent.Q(d, w) * other.vd[d][w])[
                    0
                ]
        return r

    def subs(self, subs_dict:dict)->Self:
        return self.xreplace(subs_dict)

    def xreplace(self, subs_dict:dict):
        nvd = {d: copy.copy(self.vd[d]) for d in self.vd}
        for d in nvd:
            for w in nvd[d]:
                nvd[d][w] = nvd[d][w].xreplace(subs_dict)
        return self.parent.elt(nvd)

    def large_subs(self, subs_dict):
        # # To Do
        # cd=copy.copy(self.coeff_dict)
        # for k in cd:
        #     IF=Indexed_factors(cd[k])
        #     NS={}
        #     for A in IF:
        #         if A in subs_dict: NS[A]=subs_dict[A]
        #     cd[k]=cd[k].subs(NS)
        # return self.parent.cochain(cd)
        NotImplemented

    def update_add(self, d:int, w:int, i:int, c:sp.Expr)->None:
        """Adds c times the specified basis element to self
        INPUTS:
        * 'd' = degree
        * 'w' = weight
        * 'i' = index
        * 'c' = coefficient
        """
        if d not in self.vd:
            self.vd[d] = {}
        if w not in self.vd[d]:
            self.vd[d][w] = sp.SparseMatrix(sp.zeros(len(self.parent.basis(d, w)), 1))
        self.vd[d][w][i] = self.vd[d][w][i] + c

    def update_add_vec(self, d:int, w:int, v:VectorInput):
        """Adds vector v to the specified degree and weight of self
        INPUTS:
        * 'd' - degree
        * 'w' - weight
        * 'v' - vector
        """
        v_mat=to_sympy_matrix(v)
        if d not in self.vd:
            self.vd[d] = {}
        if w not in self.vd[d]:
            self.vd[d][w] = sp.SparseMatrix(v_mat)
        self.vd[d][w] = sp.SparseMatrix(self.vd[d][w] + v_mat)

    def update_add_dict(self, ovd:dict[int,dict[int,sp.Matrix]]):
        """Adds the vector dict ovd to self
        INPUTS:
        * 'ovd' - a vector dict
        """
        for d in ovd:
            for w in ovd[d]:
                self.update_add_vec(d, w, ovd[d][w])
