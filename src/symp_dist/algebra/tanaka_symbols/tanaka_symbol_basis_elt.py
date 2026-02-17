from __future__ import annotations
from typing import TYPE_CHECKING

__all__ = ["TSymbBasisElt"]
from ..tanaka_symbols import TSymbElt
import sympy as sp
from ...utils.math_helpers import sort_basis_tuple
import copy
from ...utils.exceptions import invalid_parent_exception

if TYPE_CHECKING:
    from ..tanaka_symbols import TSymb

class TSymbBasisElt(TSymbElt):
    def __init__(self, str_rep:str, symb_str_rep:str, wght:int, parent:TSymb, i:int, admat:sp.MatrixBase):
        self.str_rep:str = str_rep
        self.symb_str_rep:str = symb_str_rep
        self.wght:int = wght
        tvec:sp.MatrixBase = sp.SparseMatrix([0] * (len(parent.basis_strs)))
        tvec[i] = 1

        super().__init__(tvec, parent)
        self.ad_mat_cache:dict = {"g": sp.SparseMatrix(admat)}
        self.length = None
        if parent.Q is not None:
            i = parent.basis_strs.index(str_rep)
            self.length = parent.Q[i, i]
        self.scalar_cache:dict[str,dict[int,dict[int,sp.MatrixBase]]] = {"CE": {}, "E": {}} # In the case of the symplectic symbol, g_0 acts diagonally

    def __str__(self):
        return self.str_rep

    def __repr__(self):
        return self.str_rep

    def __lt__(self, other):
        if self.parent != other.parent:
            raise invalid_parent_exception
        return self.parent.basis.index(self) < self.parent.basis.index(other)

    def __gt__(self, other):
        if self.parent != other.parent:
            raise invalid_parent_exception
        return self.parent.basis.index(self) > self.parent.basis.index(other)

    def __le__(self, other):
        if self.parent != other.parent:
            raise invalid_parent_exception
        return self.parent.basis.index(self) <= self.parent.basis.index(other)

    def __ge__(self, other):
        if self.parent != other.parent:
            raise invalid_parent_exception
        return self.parent.basis.index(self) >= self.parent.basis.index(other)

    def symb_expr(self):
        return sp.symbols(self.symb_str_rep)

    def ad_mat(self, mod="g", d=None, w=None, filtered=False):
        """Returns the mat rep of ad(self) on mod, either as a matrix
        or as a dictionary with keys (target_deg, target_wght) and matrix values
        """
        if mod == "g":
            return self.ad_mat_cache["g"]  # cached upon initialization
        if mod == "g_dual":
            return -self.ad_mat("g").transpose()
        if mod == "m":
            r = copy.copy(self.ad_mat("g"))
            for i in reversed(list(range(len(self.parent.basis)))):
                if self.parent.basis[i].wght >= 0:
                    r.col_del(i)
                    r.row_del(i)
            return r
        if mod == "m_dual":
            return -self.ad_mat("m").transpose()

        if d is None or w is None:
            print("ad mat requires d and w args for tensor modules")
        if (mod, d, w) not in self.ad_mat_cache:
            self.init_ad_mat(mod, d, w)
        return self.ad_mat_cache[(mod, d, w)]

    def init_ad_mat(self, mod="g", deg=None, wght=None):
        """caches the adjoint matrix of this element acting on mod
        INPUTS:
            * "mod" -- module of action, among 'g', 'g_dual','m','m_dual,'E','CE'
            * "deg" -- degree of exterior element/cochain, if applicable
            * "wght" -- wght of exterior element/cochain, if applicable

        Importantly, m, m_dual, Emd (Exterior alg of m_dual), and CE=C(m,g) are only
        m-modules, not g-modules. Accordingly, g_+ basis elts return NONE for these modules
        """

        if mod == "E":
            if ("E", deg, wght) in self.ad_mat_cache:
                return None
            E = self.parent.ext_alg
            col_list = []

            ## Deal with len(basis)=0 cases first
            l_dom = len(self.parent.ext_alg.basis(deg, wght))
            l_codom = len(self.parent.ext_alg.basis(deg, wght + self.wght))
            if l_dom == 0 or l_codom == 0:
                self.ad_mat_cache[("E", deg, wght)] = sp.SparseMatrix(
                    sp.zeros(l_codom, l_dom)
                )
                return self.ad_mat_cache[("E", deg, wght)]

            for k in range(len(E.basis(deg, wght))):
                W = E.basis(deg, wght)[k]
                comp = W.components
                # compute ad(self)(W) as a vector
                v = sp.zeros(len(E.basis(deg, wght + self.wght)), 1)
                for i in range(deg):
                    adXi = self.ad_mat(mod="m_dual").col(
                        self.parent.m_basis.index(comp[i])
                    )
                    for j in range(len(self.parent.m_basis)):
                        if adXi[j] != 0:
                            s = [str(A) for A in comp]
                            t = s[0:i] + s[i + 1 : len(s)]
                            if self.parent.m_basis_strs[j] not in t:
                                s[i] = self.parent.m_basis_strs[j]
                                s, sgn = sort_basis_tuple(
                                    tuple(s), self.parent.m_basis_strs
                                )
                                v[E.dwi(tuple(s))[2]] += sgn * adXi[j]
                col_list.append(v)
            self.ad_mat_cache[("E", deg, wght)] = sp.SparseMatrix(
                list(map(list, col_list))
            ).transpose()

        if mod == "CE":
            if ("CE", deg, wght) in self.ad_mat_cache:
                return None
            C = self.parent.cochain_complex
            if len(C.basis(deg, wght)) == 0 or len(C.basis(deg, wght + self.wght)) == 0:
                self.ad_mat_cache[("CE", deg, wght)] = sp.zeros(
                    len(C.basis(deg, wght + self.wght)), len(C.basis(deg, wght))
                )
                return None
            col_list = []
            for k in range(len(C.basis(deg, wght))):
                W = C.basis(deg, wght)[k]
                comp = W.components
                # compute ad(self)(W) as a vector
                v = sp.zeros(len(C.basis(deg, wght + self.wght)), 1)

                # Act on the wedge(m^*) part
                for i in range(deg):
                    adXi = self.ad_mat(mod="m_dual").col(
                        self.parent.m_basis.index(comp[i])
                    )
                    for j in range(len(self.parent.m_basis)):
                        if adXi[j] != 0:
                            s = [str(A) for A in comp[0:-1]]
                            t = s[0:i] + s[i + 1 : len(s)]
                            if self.parent.m_basis_strs[j] not in t:
                                s[i] = self.parent.m_basis_strs[j]
                                s, sgn = sort_basis_tuple(
                                    tuple(s), self.parent.m_basis_strs
                                )
                                s = list(s) + [str(comp[-1])]
                                v[C.dwi(tuple(s))[2]] += sgn * adXi[j]

                # Act on the g part
                adXi = self.ad_mat().col(self.parent.basis.index(comp[-1]))
                for j in range(len(self.parent.basis)):
                    if adXi[j] != 0:
                        s = [str(A) for A in comp]
                        s[-1] = self.parent.basis_strs[j]
                        v[C.dwi(tuple(s))[2]] += adXi[j]
                col_list.append(v)
            self.ad_mat_cache[("CE", deg, wght)] = sp.SparseMatrix(
                list(map(list, col_list))
            ).transpose()

            # Set the scalar cache for wght 0 elts
            if self.wght == 0:
                if deg not in self.scalar_cache["CE"]:
                    self.scalar_cache["CE"][deg] = {}
                A = self.ad_mat("CE", deg, wght)
                self.scalar_cache["CE"][deg][wght] = sp.Matrix(
                    [A[i, i] for i in range(sp.shape(A)[0])]
                )

    def cast_as_ext_elt(self):
        return self.parent.ext_alg.elt_from_cd({(str(self),): 1})

    def cast_as_cochain(self):
        return self.parent.cochain_complex.elt_from_cd({(str(self),): 1})

    def Ad_mat(self, mod="g", d=None, w=None, filtered=False):
        """Returns exp(Ad(self)) for the given module; assumes coordinates of the first kind"""
        return sp.exp(self.ad_mat(mod))
