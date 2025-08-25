from __future__ import annotations
from typing import TYPE_CHECKING

__all__ = ["SympSymb"]

from ..tanaka_symbols.tanaka_symbol import TSymb
from ...utils.exceptions import heis_dim_exception
import sympy as sp

if TYPE_CHECKING:
    from ...utils.types import VectorInput
    from ..tanaka_symbols.tanaka_symbol_elt import TSymbElt


class SympSymb(TSymb):
    def __init__(self, heis_dim: int, co_prolongation: int=0):
        if heis_dim < 3 or heis_dim % 2 == 0:
            raise heis_dim_exception
        self.heis_dim = heis_dim
        self.basis_strs = ["Y", "H", "E", "X"]
        for i in range(1, heis_dim):
            self.basis_strs.append("e_%d" % i)
        self.basis_strs.append("N")
        self.symb_str_reps = ["Y", "H", "E", "X"]
        for i in range(1, heis_dim):
            self.symb_str_reps.append("e_{" + str(i) + "}")
        self.symb_str_reps.append("N")

        self.wght_list = [1, 0, 0, -1]
        for i in range(4, len(self.basis_strs) - 1):
            self.wght_list.append(-i + 3 + co_prolongation)
        self.wght_list.append(-heis_dim + 2 * co_prolongation)
        Q = sp.SparseMatrix(
            sp.diag(
                *[1, 2, 2, 1]
                + [
                    sp.Rational(sp.factorial(i - 1),sp.factorial(heis_dim - i - 1))
                    for i in range(1, heis_dim)
                ]
                + [1]
            )
        )

        super().__init__(
            self.basis_strs, self.symb_str_reps, self.wght_list, self.ad_mats(), Q
        )

    def elt(self, vec:VectorInput=None):
        from .symplectic_symbol_elt import SympSymbElt
        return SympSymbElt.from_tsymb_elt(super().elt(vec))

    def ad_mats(self)->list[sp.MatrixBase]:
        # I'll try to avoid using this
        # Following Medvedev's basis/conventions, except the error in [Y,e_i]=(i-1)*(2*m+1-i)e_{i-1}
        basis_len = len(self.basis_strs)
        Y = sp.SparseMatrix(sp.zeros(basis_len))
        Y[0, 1] = 2
        Y[1, 3] = -1
        for i in range(2, basis_len - 3):
            Y[i + 2, i + 3] = (i - 1) * (basis_len - 4 - i)

        H = sp.SparseMatrix(sp.zeros(basis_len))
        H[0, 0] = -2
        H[3, 3] = 2
        for i in range(1, basis_len - 4):
            H[i + 3, i + 3] = 2 * i - basis_len + 4

        X = sp.SparseMatrix(sp.zeros(basis_len))
        X[1, 0] = 1
        X[3, 1] = -2
        for i in range(1, basis_len - 5):
            X[i + 4, i + 3] = 1

        E = sp.SparseMatrix(sp.zeros(basis_len))
        for i in range(1, basis_len - 4):
            E[i + 3, i + 3] = 1
        E[-1, -1] = 2

        ei = [
            sp.SparseMatrix(sp.zeros(basis_len)) for i in range(basis_len - 4)
        ]

        for i in range(2, len(ei)):
            ei[i][2 + i, 0] = -(i - 1) * (basis_len - 4 - i)
        for i in range(1, len(ei)):
            ei[i][3 + i, 1] = -(2 * i - basis_len + 4)
            ei[i][3 + i, 2] = -1
            ei[i][-1, basis_len - i - 1] = (-1) ** i
        for i in range(1, len(ei) - 1):
            ei[i][4 + i, 3] = -1

        N = sp.SparseMatrix(sp.zeros(basis_len))
        N[-1, 2] = -2

        return [Y, H, E, X] + ei[1 : len(ei)] + [N]

    ## Below this point, the methods are specific to the prolonged symbol for the symplectification

    def first_kind_to_second(self, g_elt:TSymbElt)->TSymbElt:
        """Converts an element of G_+ represented in canonical coords of the first kind
        (i.e., exp(eE+hH+yY)) to its representation in canonical coords of the second
        kind (i.e., exp(eE+hH)exp(yY)

        INPUTS:
        * 'v' -- an element of the positive part of self.alg
        """
        B = self.basis
        v = g_elt.vec
        if v[1] != 0:
            return (
                (v[0] * (sp.exp(2 * v[1]) - 1) / (2 * v[1])) * B[0]
                + v[1] * B[1]
                + v[2] * B[2]
            )
        return v[0] * B[0] + v[1] * B[1] + v[2] * B[2]

    def second_kind_to_first(self, g_elt:TSymbElt)->TSymbElt:
        """Converts an element of G_+ represented in canonical coords of the
        second kind (i.e., exp(eE+hH)exp(yY) to its representation in canonical
        coords of the first kind (i.e., exp(eE+hH+yY))

        INPUTS:
        * 'g_elt' -- an element of the positive part of self.alg
        """
        v = g_elt.vec
        B = self.basis
        if v[1] != 0:
            return (
                2 * v[1] * v[0] / (sp.exp(2 * v[1]) - 1) * B[0]
                + v[1] * B[1]
                + v[2] * B[2]
            )
        return v[0] * B[0] + v[1] * B[1] + v[2] * B[2]

    def second_kind_inv(self, g_elt:TSymbElt)->TSymbElt:
        """Returns the inverse of and elt of G_+ represented in canonical coordinates
        of the second kind.

        INPUTS:
        * 'g_elt' -- an element of the positive part of self.alg
        """
        B = self.basis
        v = g_elt.vec
        return -sp.exp(-2 * v[1]) * v[0] * B[0] - v[1] * B[1] - v[2] * B[2]

    def second_kind_mul(self, g_elt_1:TSymbElt, g_elt_2:TSymbElt)->TSymbElt:
        """Returns g_elt_1*g_elt_2, which is an elt of G_+ represented in canonical coordinates
        of the second kind

        INPUTS:
        * 'g_elt_1','g_elt_2' -- elements of the positive part of self.alg
        """
        B = self.basis
        v1 = g_elt_1.vec
        v2 = g_elt_2.vec
        return (
            (sp.exp(2 * v2[1]) * v1[0] + v2[0]) * B[0]
            + (v1[1] + v2[1]) * B[1]
            + (v1[2] + v2[2]) * B[2]
        )
