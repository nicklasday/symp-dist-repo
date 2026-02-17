__all__ = ["DiffOpDict"]

import sympy as sp
from numbers import Number
from ..algebra.tanaka_symbols import TSymbElt,TSymb
from ..algebra.complexes.cochain import Cochain


class DiffOpDict:
    """A DiffOpDict object represents a differential operator which is a
    composition of fundamental derivatives with coefficients from the curvature tensor
    of the RegularCartanGeometry and its derivatives
    """

    def __init__(self, d:dict[tuple[int,...],sp.Expr], symb:TSymb, curv:Cochain):
        """Initializes a DiffOpDict from the dictionary d on the RegularCartanGeometry P.
        The keys of d should be tuples of indices, indicating fundamental derivatives; the values
        of d are the corresponding coefficients."""
        self.symb:TSymb = symb
        self.curv:Cochain = curv
        self.d:dict[tuple[int,...],sp.Expr] = d

    def __repr__(self)->str:
        return str(self.d)

    def __str__(self)->str:
        return str(self.d)

    def __add__(self, other)->'DiffOpDict':
        # Perhaps parent check should be added to these methods
        d = {}
        for k in set(self.d.keys()).union(set(other.d.keys())):
            d[k] = 0
            if k in other.d:
                d[k] += other.d[k]
            if k in self.d:
                d[k] += self.d[k]
        return DiffOpDict(d, self.symb, self.curv)

    def __mul__(self, other)->'DiffOpDict':
        if type(other) == DiffOpDict:
            r = DiffOpDict({}, self.symb, self.curv)
            for t1 in self.d:
                t1_term = DiffOpDict(other.d, self.symb, self.curv)
                for i in reversed(t1):
                    temp = DiffOpDict({}, self.symb, self.curv)
                    for t2 in t1_term.d:
                        coeff = DiffOpDict({(i,): 1}, self.symb, self.curv).apply(
                            t1_term.d[t2]
                        )
                        temp += DiffOpDict(
                            {(i,) + t2: t1_term.d[t2]}, self.symb, self.curv
                        )
                        if coeff != 0:
                            temp += DiffOpDict({t2: coeff}, self.symb, self.curv)
                    t1_term = temp
                r += self.d[t1] * t1_term
            return r
        else:
            r_dict = {}
            for k in self.d:
                r_dict[k] = other * self.d[k]
            return DiffOpDict(r_dict, self.symb, self.curv)

    def __rmul__(self, other)->'DiffOpDict':
        # We shouldn't arrive here if other is a DiffOpDict
        return self * other

    def __neg__(self)->'DiffOpDict':
        d = {}
        for k in self.d:
            d[k] = -self.d[k]
        return DiffOpDict(d, self.symb, self.curv)

    def __sub__(self, other)->'DiffOpDict':
        return self + (-other)

    def __eq__(self, other)->bool:
        if type(other) == DiffOpDict:
            z = self - other
            z.clear_zeros()
            return z == 0
        self.clear_zeros()
        if set(self.d.keys()) == {tuple()}:
            return other == self.d[tuple()]
        if set(self.d.keys()) == set():
            return other == 0
        return False

    def clear_zeros(self)->None:
        for k in set(self.d.keys()):
            if self.d[k] == 0:
                self.d.pop(k)

    def weakly_normalize_tuple(self, t:tuple[int,...])->'DiffOpDict':
        """Returns a weakly normal DiffOpDict which is equivalent to t. Weakly normal
        means only indices of weight greater than -2 are included in self
        
        Note: This only works for distributions with the symplectic symbol"""
        B = self.symb.basis
        a = None
        for i in reversed(t):
            if B[i].wght < -1:
                a = i
        if a is None:
            return DiffOpDict({t: 1}, self.symb, self.curv)

        # Find Xi,Xj so that [Xi,Xj]=B[a]. 
        # We aren't guaranteed this exists for arbitrary nilp. Lie algs.
        # I've cut a corner here; this won't work for non-symplectic symbols
        Xi = None
        Xj = None
        for i in range(len(B)):
            for j in range(i, len(B)):
                if B[i].ad(B[j]) == B[a]:
                    Xi, Xj = [B[i], B[j]]
        i, j = [B.index(Xi), B.index(Xj)]

        pre = DiffOpDict({t[0 : t.index(a)]: 1}, self.symb, self.curv)
        post = DiffOpDict({t[t.index(a) + 1 : len(t)]: 1}, self.symb, self.curv)
        r = pre * DiffOpDict({(i, j): 1}, self.symb, self.curv) * post
        r += -pre * DiffOpDict({(j, i): 1}, self.symb, self.curv) * post

        if Xi is None or Xj is None:
            raise NotImplementedError
        w = Xi.cast_as_ext_elt().wedge(Xj.cast_as_ext_elt())
        im_elt = self.curv.apply_cochain_map(w)
        for k in range(len(B)):
            if im_elt.vec[k] != 0:
                r += (
                    pre * DiffOpDict({(k,): im_elt.vec[k]}, self.symb, self.curv) * post
                )
        return r.weakly_normal_form()

    def strongly_normalize_tuple(self, t:tuple[int,...])->'DiffOpDict':
        """Returns a normal DiffOpDict which is equivalent to t. Strongly normal means
        only indices of weight greater than -2 are included in self, and that the
        nonnegative operators appear first"""
        if len(t) == 1:
            return DiffOpDict({t: 1}, self.symb, self.curv)
        B = self.symb.basis
        a = None
        for i in range(len(t) - 1):
            if B[t[i]].wght >= 0 and B[t[i + 1]].wght < 0:
                a = i
        if a is None:
            return DiffOpDict({t: 1}, self.symb, self.curv)

        pre = DiffOpDict({t[0:a]: 1}, self.symb, self.curv)
        post = DiffOpDict({t[a + 2 : len(t)]: 1}, self.symb, self.curv)
        r = pre * DiffOpDict({(t[a + 1], t[a]): 1}, self.symb, self.curv) * post
        im_elt = B[t[a]].ad(B[t[a + 1]])
        for i in range(len(B)):
            if im_elt.vec[i] != 0:
                r += (
                    pre * DiffOpDict({(i,): im_elt.vec[i]}, self.symb, self.curv) * post
                )
        return r.normal_form()

    def weakly_normal_form(self)->'DiffOpDict':
        """Returns the weakly normal form of self"""
        # if self.query("WeaklyNormal"): return DiffOpDict(self.d,self.symb, self.curv)
        r = DiffOpDict({}, self.symb, self.curv)
        for t in self.d:
            r += self.d[t] * self.weakly_normalize_tuple(t)
        r.clear_zeros()
        return r

    def normal_form(self)->'DiffOpDict':
        """Returns the normal form of self"""
        # if self.query("Normal"): return DiffOpDict(self.d,self.symb, self.curv)
        w = DiffOpDict({}, self.symb, self.curv)
        for k in self.d:
            w += self.weakly_normalize_tuple(k) * self.d[k]
        nf = DiffOpDict({}, self.symb, self.curv)
        for k in w.d:
            nf += self.strongly_normalize_tuple(k) * w.d[k]
        nf.clear_zeros()
        return nf

    def apply(self, f:sp.Expr, normal:bool=False)->sp.Expr:
        """Applies the operator represented by self to the function f, which should be an
        elementary function in terms of the curvatures of self.geom"""
        if type(f) == TSymbElt:
            return f.parent.elt([self.apply(a) for a in f.vec])
        if normal:
            nf = self
        else:
            nf = self.normal_form()
        r = 0
        for k in nf.d:
            temp = f
            for i in reversed(k):
                temp = self.normal_ind_der(temp, i)
            r += temp * nf.d[k]
        return r

    def normal_ind_der(self, f:sp.Expr, i:int)->sp.Expr:
        """Applies the fundamental derivative corresponding to i to the function

        Arguments:
            * 'i' - an integer corresponding to a fundamental derivative of weight >=-1
            * 'f' - an elementary function in terms of the curvatures of self.curv"""

        if isinstance(f, Number):
            return 0
        if isinstance(f,sp.Add):
            return sp.Add(*[self.normal_ind_der(A, i) for A in f.args])
        if isinstance(f,sp.Mul):
            result = 0
            for j in range(len(f.args)):
                result += self.normal_ind_der(f.args[j], i) * sp.Mul(
                    *list(f.args[0:j] + f.args[j + 1 : len(f.args)])
                )
            return result
        if isinstance(f,sp.Pow):
            return f.exp * f.base ** (f.exp - 1) * self.normal_ind_der(f.base, i)
        if isinstance(f,sp.Symbol):
            return 0
        if isinstance(f,sp.exp):
            return f * self.normal_ind_der(f.args[0], i)
        if isinstance(f,sp.Indexed):
            if self.symb.basis[i].wght < 0:
                return f.base[f.indices + (i,)]
            else:
                # nonnegative indices act via (-ad) after normalizing
                if len(f.indices) == 3:
                    g = self.symb
                    Xi = g.basis[i]
                    Xi_curv = (-Xi).ad(self.curv, mod="CE")
                    dwi = g.cochain_complex.dwi(
                        tuple([g.basis_strs[a] for a in f.indices])
                    )
                    try:
                        return Xi_curv.vd[dwi[0]][dwi[1]][dwi[2]]
                    except:
                        return 0
                else:
                    t = (i,) + tuple(reversed(f.indices[3 : len(f.indices)]))
                    return DiffOpDict({t: 1}, self.symb, self.curv).apply(
                        f.base[*f.indices[0:3]]
                    )

    def query(self, property:str)->bool|None:
        """Queries if self satisfies property among "WeaklyNormal" and "Normal" """
        if property == "WeaklyNormal":
            for k in self.d:
                for i in k:
                    if self.symb.basis[i].wght < -1:
                        return False
            return True
        if property == "Normal":
            if not self.query("WeaklyNormal"):
                return False
            for k in self.d:
                a = len(k)
                for i in range(len(k)):
                    if self.symb.basis[len(k) - 1 - i].wght > -1:
                        a = len(k) - 1 - i
                for i in range(a + 1, len(k)):
                    if self.symb.basis[k[i]].wght == -1:
                        return False
            return True
        return None