
__all__ = ["TensorAlgBasisElt"]

from .tensor_alg_elt import TensorAlgElt
from ...utils import math_helpers as mh

class TensorAlgBasisElt(TensorAlgElt):
    def __init__(self, parent, deg, wght, vd, str_rep):
        self.deg = deg
        self.wght = wght
        self.str_rep = str_rep
        TensorAlgElt.__init__(self, parent, vd)
        alg = parent.alg
        self.components = [alg.basis[alg.basis_strs.index(A)] for A in str_rep]

    def __str__(self)->str:
        return mh.tuple_to_str(self.str_rep)

    def __repr__(self)->str:
        return self.__str__()
