from __future__ import annotations
from typing import TYPE_CHECKING

__all__ = ["ExtElt"]

from ..tensor_algebras import TensorAlgElt
from ...utils.exceptions import invalid_parent_exception
from ..complexes import Cochain
from typing import Any

if TYPE_CHECKING:
    from ..tanaka_symbols import TSymbElt
    from .exterior_alg import ExtAlg
    from ...utils.types import VectorInput



class ExtElt(TensorAlgElt):
    def __init__(self, parent:ExtAlg, vd:dict[int, dict[int,VectorInput]]={}):
        TensorAlgElt.__init__(self, parent, vd)
        self.parent:ExtAlg

    @classmethod
    def from_cd(cls, parent:ExtAlg, cd:dict[tuple[str,...],Any]={})->'ExtElt':
        """Constructs an exterior element in parent from a coefficient dictionary"""
        return parent.elt(parent.cd_to_vd(cd))

    def wedge(self, other:'ExtElt | Cochain')->'ExtElt':
        """ Returns the wedge product of self and other
        INPUTS:
        * 'other' - another exterior element or a cochain
        
        NOTE: Since it's ambiguous whether an algebra element is dual or not, 
              other cannot be of type T_symb_elt
        """
        from ..exterior_algebras import ExtAlg
        from ..complexes import CochainComplex

        obj2_type = None
        if isinstance(other.parent, ExtAlg):
            obj2_type = "ext_elt"
            r = self.parent.elt({})
        if isinstance(other.parent, CochainComplex):
            obj2_type = "cochain"
            r = self.parent.alg.cochain_complex.elt({})
        if obj2_type is None:
            raise invalid_parent_exception(
                "wedge recieved arguments with incompatible parents"
            )

        L1 = []
        for d1 in self.vd:
            for w1 in self.vd[d1]:
                for i1 in range(len(self.vd[d1][w1])):
                    if self.vd[d1][w1][i1] != 0:
                        L1.append((d1, w1, i1, self.vd[d1][w1][i1]))
        L2 = []
        for d2 in other.vd:
            for w2 in other.vd[d2]:
                for i2 in range(len(other.vd[d2][w2])):
                    if other.vd[d2][w2][i2] != 0:
                        L2.append((d2, w2, i2, other.vd[d2][w2][i2]))
        for t1 in L1:
            d1, w1, i1, c1 = t1
            for t2 in L2:
                d2, w2, i2, c2 = t2
                d3, w3, i3, s = self.parent.wedge_indices(
                    d1, w1, i1, d2, w2, i2, obj2_type
                )
                if s != 0:
                    r.update_add(d3, w3, i3, s * c1 * c2)
        return r

    def tensor(self, v:TSymbElt)->Cochain:
        """Returns self otimes v, an element of the Chevalley-Eilenberg Complex
        INPUTS:
        * 'v' - an element of the Lie algebra of self
        """
        r = self.parent.alg.cochain_complex.elt({})
        for d in self.vd:
            for w in self.vd[d]:
                for i in range(len(self.vd[d][w])):
                    if self.vd[d][w][i] != 0:
                        r = r + self.vd[d][w][i] * self.parent.basis(d, w)[i].tensor(v)
        return r

    def symb_expr(self):
        expr = 0
        for d in self.vd:
            for w in self.vd[d]:
                for i in range(len(self.vd[d][w])):
                    if self.vd[d][w][i] != 0:
                        expr += self.vd[d][w][i] * self.parent.basis(d, w)[i].symb
        return expr
