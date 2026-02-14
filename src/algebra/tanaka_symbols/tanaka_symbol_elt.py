from __future__ import annotations
from typing import TYPE_CHECKING
__all__ = ["TSymbElt"]

import sympy as sp
from ...utils import math_helpers as mh
from ...utils import lin_alg_helpers as lh
from ...utils.exceptions import constr_Mat_size_exception, invalid_ext_elt_casting

if TYPE_CHECKING:
    from ..complexes.cochain_complex import CochainComplex
    from ..tanaka_symbols import TSymb
    from ..exterior_algebras.exterior_alg import ExtAlg
    from ...utils.types import VectorInput

class TSymbElt:
    parent: TSymb
    vec: sp.Matrix
    ad_mat_cache: dict
    ext_alg: ExtAlg
    cochain_complex: CochainComplex

    def __init__(self, vec: VectorInput, parent: TSymb):
        """vec: a list of length len(self.parent.basis) with integer entries"""
        from ..complexes.cochain_complex import CochainComplex
        from ...utils.sympy_utils import to_sympy_matrix
        self.parent = parent
        self.vec = to_sympy_matrix(vec)
        if sp.shape(self.vec)[1] != 1:
            self.vec = self.vec.transpose()  # column vector
        if sp.shape(self.vec)[0] != len(parent.basis_strs):
            raise constr_Mat_size_exception()
        self.ad_mat_cache = {}
        # To Do: Introduce filtered base here

    def __str__(self):
        return mh.str_from_vec(self.vec, self.parent.basis)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if other == 0 and type(other) == int:
            return self.vec == sp.Matrix([0] * (len(self.parent.basis)))
        if not hasattr(other, "parent"):
            return False
        if self.parent != other.parent:
            return False
        return self.vec == other.vec

    def __neg__(self):
        return TSymbElt([-A for A in self.vec], self.parent)

    def __add__(self, other):
        if other == 0:
            return self
        return TSymbElt(self.vec + other.vec, self.parent)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        return TSymbElt([other * A for A in self.vec], self.parent)

    def __rmul__(self, other):
        return self * other

    def Ad_mat(self, mod="g", d=None, w=None, filtered=False):
        """Returns the Adjoint sp.Matrix of self acting on mod
        INPUTS:
            * "mod" -- module of action, among 'g', 'g_dual','m','m_dual,'E','CE'
            * "d" -- degree of exterior element/cochain, if applicable
            * "w" -- weight of exterior element/cochain, if applicable

        Importantly, m, m_dual, Emd (Exterior alg of m_dual), and CE=C(m,g) are only
        m-modules, not g-modules. Accordingly, g_+ basis elts return NONE for these modules

        If mod is among 'E' or 'CE' the result is a dictionary with keys (td,tw), target deg and wght
        """
        if mod in ["E", "CE"]:
            raise NotImplementedError()
        if mod in ["g", "g_dual", "m", "m_dual"]:
            if d is not None or w is not None:
                raise NotImplementedError()
            return sp.exp(self.ad_mat(mod))

    def ad_mat(self, mod="g", d=None, w=None, filtered=False):
        """Returns the adjoint sp.Matrix of self acting on mod, or a dict of such matrices
        INPUTS:
            * "mod" -- module of action, among 'g', 'g_dual','m','m_dual,'E','CE'
            * "d" -- degree of exterior element/cochain, if applicable
            * "w" -- weight of exterior element/cochain, if applicable

        Importantly, m, m_dual, Emd (Exterior alg of m_dual), and CE=C(m,g) are only
        m-modules, not g-modules. Accordingly, g_+ basis elts return NONE for these modules

        If mod is among 'E' or 'CE' the result is a dictionary with keys (td,tw), target deg and wght
        """
        if mod in ["E", "CE"]:
            raise NotImplementedError()
            # I shouldn't need this

        if mod in ["g", "g_dual", "m", "m_dual"]:
            if d is not None or w is not None:
                raise NotImplementedError()
            r = sp.SparseMatrix(
                sp.zeros(self.parent.mod_dim(mod))
            )  # a sp.Matrix of appropriate dim
            for i in range(len(self.parent.basis)):
                if type(self.vec[i]) == type(sp.IndexedBase("K")[0]):
                    for j in range(sp.shape(self.parent.basis[i].ad_mat(mod))[0]):
                        for k in range(sp.shape(self.parent.basis[i].ad_mat(mod))[1]):
                            r[j, k] += (
                                self.vec[i] * self.parent.basis[i].ad_mat(mod)[j, k]
                            )
                else:
                    r += self.vec[i] * self.parent.basis[i].ad_mat(mod)
            return r

    def ad(self, t, mod="g"):
        """Returns an object representing ad(self,t), where t has constant coefficients
        INPUTS:
        * 't' - an element of mod with constant coefficients (i.e., no symbols or indexed objects)
        * 'mod' - among 'g','m','g_dual','m_dual','E,', and 'C'
        """
        P = self.parent
        E = self.parent.ext_alg
        C = self.parent.cochain_complex

        if t.parent == E:
            mod = "E"
        if t.parent == C:
            mod = "CE"

        if mod in ["E", "CE"]:
            r:dict[int,dict[int,sp.MatrixBase]] = {}
            for sd in t.vd:  # source deg
                for sw in t.vd[sd]:  # source wght
                    for i in range(len(P.basis)):  # entries of self
                        if self.vec[i] != 0:
                            td = sd  # target deg
                            tw = sw + P.basis[i].wght  # target weight
                            im_vec = (
                                self.vec[i]
                                * P.basis[i].ad_mat(mod, sd, sw)
                                * t.vd[sd][sw]
                            )
                            if td not in r:
                                r[td] = {}
                            if tw not in r[td]:
                                r[td][tw] = sp.zeros(len(t.parent.basis(td, tw)), 1)
                            r[td][tw] += im_vec
            if mod == "E":
                return E.elt(r)
            if mod == "CE":
                return C.elt(r)

        if mod in ["g", "g_dual"]:
            return t.parent.elt(self.ad_mat(mod) * t.vec)
        if mod in ["m", "m_dual"]:
            # # This isn't correct anymore; To do: Fix hardcoding here
            t_vec = t.vec
            for i in reversed(range(len(self.parent.basis))):
                if self.parent.basis[i].wght >= 0:
                    print("deleting", i)
                    t_vec.row_del(i)
            temp = self.ad_mat(mod) * t_vec
            for i in range(len(self.parent.basis)):
                if self.parent.basis[i].wght >= 0:
                    print("inserting", i)
                    t_vec = t_vec.row_insert(i, sp.zeros(1))
            print("list(temp) =", list(temp))
            return t.parent.elt(list(temp))

    def __Ad_wght_0(self, t):
        """Return"""
        P = self.parent
        r:dict[int,dict[int,sp.MatrixBase]] = {}

        for d in t.vd:  # source deg
            if d not in r:
                r[d] = {}
            for w in t.vd[d]:  # source wght
                exp_list = [0] * len(t.vd[d][w])
                for i in range(len(P.basis)):
                    if P.basis[i].wght == 0:
                        P.basis[i].init_ad_mat("CE", d, w)
                        for j in range(len(exp_list)):
                            exp_list[j] += (
                                self.vec[i] * P.basis[i].scalar_cache["CE"][d][w][j]
                            )
                coeff_list = [sp.exp(A) for A in exp_list]
                r[d][w] = sp.Matrix(
                    [[t.vd[d][w][i] * coeff_list[i]] for i in range(len(coeff_list))]
                )
        return self.parent.cochain_complex.elt(r)

    def __Ad_wght_1(self, t, mod="CE"):
        P = self.parent
        C = self.parent.cochain_complex
        r:dict[int,dict[int,sp.MatrixBase]] = {}
        # scrub t of sp.shape (0,0) entries
        for d in t.vd:
            for w in list(t.vd[d].keys()):
                if sp.shape(t.vd[d][w])[1] == 0:
                    t.vd[d].pop(w)

        for d in t.vd:
            r[d] = {}
            for w in t.vd[d]:
                r[d][w] = t.vd[d][w]
        for d in t.vd:  # source deg
            for w in t.vd[d]:  # source wght
                for i in range(len(P.basis)):  # entries of self
                    if self.vec[i] != 0 and P.basis[i].wght == 1:
                        dom_vec = t.vd[d][w]
                        sw, tw = (
                            w,
                            w + P.basis[i].wght,
                        )  # source and target deg and wght
                        if d not in r:
                            r[d] = {}
                        k = 1
                        while dom_vec != sp.zeros(*sp.shape(dom_vec)):
                            im_vec = (
                                self.vec[i] * P.basis[i].ad_mat(mod, d, sw) * dom_vec
                            ) / k
                            if tw not in r[d]:
                                r[d][tw] = sp.zeros(len(C.basis(d, tw)), 1)
                            r[d][tw] += im_vec

                            dom_vec = im_vec
                            sw, tw = (sw + P.basis[i].wght, tw + P.basis[i].wght)
                            k += 1
        return C.elt(r)

    def Ad(self, t, mod="g", coord="second kind"):
        """For the cochain complex, this returns an object representing Ad(exp(w0)exp(w1),t) if coord=='second kind',
        where t has constant coefficients and w0+w1=self is the weight decomposition of self.
        If coord=='first kind', this returns exp(ad(self))

           INPUTS:
           * 't' - an element of mod with constant coefficients
           * 'mod' - among 'g','m','g_dual','m_dual','E,', and 'C'
           * 'coord' - among 'first kind' and 'second kind'
        """
        from ...algebra.tanaka_symbols.symplectic_symbol import SympSymb
        
        E = self.parent.ext_alg
        C = self.parent.cochain_complex

        if t.parent == E:
            mod = "E"
        if t.parent == C:
            mod = "CE"

        if mod == "CE":
            if coord == "first kind":
                raise NotImplementedError()  # I shouldn't need this, really
            return self.__Ad_wght_0(self.__Ad_wght_1(t))
        if mod in ["g", "g_dual", "m", "m_dual"]:
            if coord == "first kind":
                return t.parent.elt(self.Ad_mat(mod) * t.vec)
            if coord == "second kind":
                if isinstance(self.parent,SympSymb):
                    return self.parent.second_kind_to_first(self).Ad(t,mod,"first kind")
                raise NotImplementedError 

    def iprod(self, other:TSymbElt):
        """Returns the inner product of self and other
        INPUTS:
        * 'other' - another T_symb_elt
        """
        return self.parent.iprod(self, other)

    def negative_projection(self):
        v = list(self.vec)
        for i in range(len(self.vec)):
            if self.parent.basis[i].wght > -1:
                v[i] = 0
        return self.parent.elt(v)

    def cast_as_ext_elt(self):
        l = len(self.parent.basis) - len(self.parent.m_basis)
        if self.vec[0:l] != [0] * l:
            raise invalid_ext_elt_casting
        return self.parent.ext_alg.elt_from_cd(
            {
                (self.parent.basis_strs[i],): self.vec[i]
                for i in range(len(self.parent.basis))
                if self.vec[i] != 0
            }
        )

    def cast_as_cochain(self):
        return self.parent.cochain_complex.elt_from_cd(
            {
                (self.parent.basis_strs[i],): self.vec[i]
                for i in range(len(self.parent.basis))
                if self.vec[i] != 0
            }
        )
