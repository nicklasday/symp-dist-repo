from __future__ import annotations
from typing import TYPE_CHECKING

__all__ = ["Cochain"]

import sympy as sp
from ..tensor_algebras import TensorAlgElt
from ...utils.exceptions import invalid_parent_exception
from IPython.display import display

if TYPE_CHECKING:
    from ...algebra.tanaka_symbols import TSymbElt
    from ...algebra.exterior_algebras.exterior_elt import ExtElt
    from ...algebra.complexes.cochain_complex import CochainComplex
    from ...algebra.tanaka_symbols.tanaka_symbol_basis_elt import TSymbBasisElt

class Cochain(TensorAlgElt):
    def __init__(self, parent:CochainComplex, vd:dict[int,dict[int,sp.MatrixBase]]={}):
        TensorAlgElt.__init__(self, parent, vd)
        self.parent:CochainComplex

    @classmethod
    def from_cd(cls, parent:CochainComplex, cd:dict[tuple,sp.Expr]={})->'Cochain':
        """Constructs a cochain in parent from a coefficient dictionary"""
        return parent.elt(parent.cd_to_vd(cd))

    def __gt__(self, other)->bool:
        if self.parent != other.parent:
            raise invalid_parent_exception
        return str(self) > str(other)

    def __ge__(self, other)->bool:
        if self.parent != other.parent:
            raise invalid_parent_exception
        return str(self) >= str(other)

    def __lt__(self, other)->bool:
        if self.parent != other.parent:
            raise invalid_parent_exception
        return str(self) < str(other)

    def __le__(self, other)->bool:
        if self.parent != other.parent:
            raise invalid_parent_exception
        return str(self) <= str(other)

    def symb_expr(self)->sp.Expr:
        expr = 0
        for d in self.vd:
            for w in self.vd[d]:
                for i in range(len(self.vd[d][w])):
                    if self.vd[d][w][i] != 0:
                        expr += self.vd[d][w][i] * self.parent.basis(d, w)[i].symb
        return expr

    def pprint(self)->None:
        expr = 0
        for d in self.vd:
            for w in self.vd[d]:
                for i in range(len(self.vd[d][w])):
                    if self.vd[d][w][i] != 0:
                        expr += self.vd[d][w][i] * self.parent.basis(d, w)[i].symb
        display(sp.simplify(expr))

    def cb(self)->'Cochain':
        """Returns the coboundary map of C(m,g) applied to c"""
        r:dict[int,dict[int,sp.MatrixBase]] = {}
        for d in self.vd:
            for w in self.vd[d]:
                if sp.shape(self.parent.cb_mat(d, w))[1] == 0:
                    v = sp.zeros(len(self.parent.basis(d + 1, w)), 1)
                else:
                    v = self.parent.cb_mat(d, w) * self.vd[d][w]
                if d + 1 not in r:
                    r[d + 1] = {}
                if w not in r[d + 1]:
                    r[d + 1][w] = sp.SparseMatrix(
                        sp.zeros(len(self.parent.basis(d + 1, w)), 1)
                    )
                r[d + 1][w] = r[d + 1][w] + v
        return self.parent.elt(r)

    def cb_preim_elt(self,check=False)->'Cochain':
        """Returns a cochain which maps to self under the coboundary.
        If self is not exact, returns None.
        """
        return self.parent.cb_preim_elt(self,check)

    def check_valid_vd(self)->bool:
        """Returns True if self has a vector dictionary which is a
        valid representation of a cochain from self.parent, False otherwise
        """
        for d in self.vd:
            for w in self.vd[d]:
                if sp.shape(self.vd[d][w]) != (len(self.parent.basis(d, w)), 1):
                    return False
        return True

    def wght_proj(self, w:int)->'Cochain':
        """returns a cochain representing the projection of self onto weight w
        INPUTS:
        * 'w' -- an integer weight
        """
        nvd = {}
        for d in self.vd:
            if w in self.vd[d]:
                nvd[d] = {w: self.vd[d][w]}
        return self.parent.elt(nvd)

    def deg_proj(self, d:int)->'Cochain':
        """returns a cochain representing the projection of self onto weight w
        INPUTS:
        * 'd' -- a nonnegative integer degree
        """
        if d not in self.vd:
            return self.parent.elt({})
        return self.parent.elt({d: self.vd[d]})

    def apply_cochain_map_base(self, *elts:TSymbBasisElt)->TSymbElt:
        """returns self(elt), taking self to be an element of Hom(Lambda^* m,g)
        INPUTS:
        * 'elt_list' - elements from the Lie algebra basis, all of negative weight
        """
        for A in elts:
            if A.wght > -1:
                print(
                    "apply_cochain_map_base only accepts arguments of negative weight"
                )

        r = self.parent.alg.elt()
        ord_strs, s = self.parent.alg.ext_alg.sort_tuple(tuple([str(A) for A in elts]))
        if ord_strs is None:
            return r
        for A in self.parent.alg.basis:
            d, w, i = self.parent.dwi(tuple(list(ord_strs) + [str(A)]))
            if d in self.vd and w in self.vd[d]:
                r += s * self.vd[d][w][i] * A
        return r

    def apply_cochain_map(self, ext_elt:ExtElt)->TSymbElt:
        """returns self(ext_elt), taking self to be an element of Hom(Lambda^* m,g)
        INPUTS:
        * 'ext_elt' -- an element of the exterior algebra of the Lie algebra
        """
        T = self.parent.alg
        E = self.parent.alg.ext_alg
        r = T.elt()
        for d in ext_elt.vd:
            for w in ext_elt.vd[d]:
                for i in range(len(ext_elt.vd[d][w])):
                    if ext_elt.vd[d][w] != 0:
                        b = E.basis(d, w)[i]
                        r += ext_elt.vd[d][w][i] * self.apply_cochain_map_base(
                            *b.components
                        )
        return r

    def one_cochain_mat_rep(self):
        return self.parent.one_cochain_mat_rep(self)
