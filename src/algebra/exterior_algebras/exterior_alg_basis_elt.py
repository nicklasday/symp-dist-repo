from __future__ import annotations
from typing import TYPE_CHECKING
__all__ = ["ExtBasisElt"]

import sympy as sp
from ...utils.types import VectorInput
from .exterior_elt import ExtElt
from ..tensor_algebras.tensor_alg_basis_elt import TensorAlgBasisElt


if TYPE_CHECKING:
    from ..tanaka_symbols import TSymbElt
    from .exterior_alg import ExtAlg


class ExtBasisElt(ExtElt, TensorAlgBasisElt):
    def __init__(self, parent:ExtAlg, deg:int, wght:int, vd:dict[int,dict[int,VectorInput]], str_rep:tuple[str,...]):
        ExtElt.__init__(self, parent, vd)
        TensorAlgBasisElt.__init__(self, parent, deg, wght, vd, str_rep)
        self.length = None
        if parent.alg.Q is not None:
            self.length = sp.Rational(1, sp.prod([A.length for A in self.components]))

        # Set the symbol representation of self
        if len(self.components) == 0:
            # To do: Adjust the print routine for degree 0 forms
            self.symb = sp.symbols("ONE")
        else:
            sl = [A.symb_str_rep for A in self.components]
            s = "{" + sl[0] + r"^*}"
            for i in range(1, len(sl)):
                s += r"\wedge{" + sl[i] + r"^*}"
            self.symb = sp.symbols(s)

    def tensor(self, v:TSymbElt):
        """Returns self otimes v, and element of the Chevalley-Eilenberg Complex
        INPUTS:
        * 'v' - an element of the Lie algebra of self
        """
        r_dict = {}
        for i in range(len(v.vec)):
            if v.vec[i] != 0:
                curr_str = tuple(
                    [str(a) for a in self.components] + [self.parent.alg.basis_strs[i]]
                )
                r_dict[curr_str] = v.vec[i]
        return self.parent.alg.cochain_complex.elt_from_cd(r_dict)
