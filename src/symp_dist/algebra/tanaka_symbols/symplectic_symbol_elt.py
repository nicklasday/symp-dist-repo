from __future__ import annotations
from typing import TYPE_CHECKING

__all__ = ["SympSymbElt"]
from .tanaka_symbol_elt import TSymbElt
from .symplectic_symbol import SympSymb
from ...utils.exceptions import invalid_parent_exception
import sympy as sp
from typing import override

if TYPE_CHECKING:
    from ..tensor_algebras import TensorAlgElt
    from ...utils.types import VectorInput


class SympSymbElt(TSymbElt):
    def __init__(self, vec:VectorInput, parent: SympSymb):
        super().__init__(vec, parent)
        self.parent:SympSymb

    @classmethod
    def from_tsymb_elt(cls, tsymb_elt:TSymbElt)->'SympSymbElt':
        if isinstance(tsymb_elt.parent,SympSymb):
            return cls(tsymb_elt.vec,tsymb_elt.parent)
        else:
            raise invalid_parent_exception

    @override   
    def Ad(self, t, mod="g", coord="second kind")->TensorAlgElt:
        """For the cochain complex, this returns an object representing Ad(exp(w0)exp(w1),t) if coord=='second kind',
        where t has constant coefficients and w0+w1=self is the weight decomposition of self.
        If coord=='first kind', this returns exp(ad(self))

           INPUTS:
           * 't' - an element of mod with constant coefficients
           * 'mod' - among 'g','m','g_dual','m_dual','E,', and 'C'
           * 'coord' - among 'first kind' and 'second kind'
        """
        P = self.parent

        if coord == "second kind" and mod in ["g", "g_dual", "m", "m_dual"]:
            return t.parent.elt(
                sp.exp(P.second_kind_to_first(self.vec).ad_mat(mod)) * t.vec
            )
        return super().Ad_mat(t,mod,coord)
