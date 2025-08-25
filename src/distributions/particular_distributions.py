__all__ = [
    "Particular_distr",
    "Weak_Single_Monge_distr",
    "Invol_6_Prenorm_distr",
    "Invol_7_Prenorm_distr",
    "Standard_Prenorm_distr",
    "Zero_Wilc_distr",
    "Weak_Prenorm_distr",
    "Weakest_Prenorm_distr",
    "Flat_distr",
    "Free_distr",
]

import sympy as sp
from sympy import IndexedBase
from ..utils import ds_helpers as dsh
from ..algebra.complexes import CochainComplex
from .distribution import Distr_of_constant_symbol, Compute_JS_dict
from ..algebra.tanaka_symbols import TSymb, SympSymb
from typing import Tuple


def Particular_distr(
    g:TSymb,
    prenorm_dict:dict[sp.Indexed,sp.Expr]={},
    Jacobi_id_wght:int=0,
    unshelve:bool=False,
    reshelve:bool=False,
    name:None|str=None,
    constant_coeff:bool=False,
)->Tuple[Distr_of_constant_symbol,dict,list[sp.Expr]]:
    """Returns a Distribution of constant symbol with the given prenormalization along with its Jacobi
    substitution dictionary.
    INPUTS:
    * 'g' - a T_symbol object"""
    K = IndexedBase("K")
    C = CochainComplex(g)
    c = C.elt({})

    # Construct a distribution D with the given Tanaka symbol and prenormalization
    for i in range(len(g.basis)):
        if g.basis[i].wght < 0:
            for j in range(i + 1, len(g.basis)):
                for k in range(len(g.basis)):
                    if (
                        g.basis[k].wght < 0
                        and g.basis[k].wght > g.basis[i].wght + g.basis[j].wght
                    ):
                        c = c + C.elt_from_cd(
                            {
                                (str(g.basis[i]), str(g.basis[j]), str(g.basis[k])): K[
                                    i, j, k
                                ]
                            }
                        )
    c = c.subs(prenorm_dict)
    temp_D = Distr_of_constant_symbol(g, c)

    # Compute the Jacobi identity dict J_subs, substitute into curvature and return
    J_subs, not_added = Compute_JS_dict(
        temp_D, Jacobi_id_wght, unshelve, reshelve, name, constant_coeff
    )
    c = dsh.ds_subs(c, J_subs)[0]
    return (Distr_of_constant_symbol(g, c), J_subs, not_added)


def Weak_Single_Monge_distr(
    g:SympSymb, Jacobi_id_wght:int=0, unshelve:bool=False, reshelve:bool=False, constant_coeff:bool=False
)->Tuple[Distr_of_constant_symbol,dict,list[sp.Expr]]:
    """This is just some of the relations one can impose upon a frame for a
    single Monge equation."""

    K = IndexedBase("K")
    s = {}
    l = len(g.basis)

    # # [X,ei] = e_{i+1} mod <e1> (for ALL i)
    for i in range(4, l - 1):
        for j in [3] + list(range(5, l)):
            s[K[3, i, j]] = 0

    # # [ei,ej] in <N> (for ALL i,j)
    for i in range(4, l - 1):
        for j in range(i + 1, l - 1):
            for k in range(3, l - 1):
                s[K[i, j, k]] = 0

    # # [X,N] in <e1,N>
    for k in range(4, l - 1):
        s[K[3, 10, k]] = 0

    # # [ei,N] in <N> (for ALL i)
    for i in range(4, l - 1):
        for j in range(3, l - 1):
            s[K[i, l - 1, j]] = 0

    # # # [e1,N]=[e2,N]=0 # This is specific to symplectified (2,6) Monge; i.e., (2,8) Monge
    # s[K[4,l-1,l-1]]=0
    # s[K[5,l-1,l-1]]=0

    # # I tried some stuff with alpha here, but it hasn't really worked out
    # temp=Particular_distr(g,s,0)[0]
    # # # [X,ei] = e_{i+1} - ei(alpha)*e1
    # alpha=IndexedBase('alpha')
    # for i in range(4,l-1):
    #     s[K[3,i,4]]=-temp.normalize_der(temp.abn_ind_der(alpha[0,0,0],i))
    return Particular_distr(
        g,
        s,
        Jacobi_id_wght,
        unshelve,
        reshelve,
        name="Weak_Single_Monge_distr" + str(len(g.basis)),
        constant_coeff=constant_coeff,
    )


def Invol_7_Prenorm_distr(
    g:SympSymb, Jacobi_id_wght:int=0, unshelve:bool=False, reshelve:bool=False, constant_coeff:bool=False
)->Tuple[Distr_of_constant_symbol,dict,list[sp.Expr]]:
    """Returns a symplectified (2,7) distr with normalization utilizing the
      involutivity conditions of symplectified distributions; that is...

    INPUTS:
    * 'g' - a symplectic Tanaka symbol"""

    K = IndexedBase("K")
    s = {}
    l = len(g.basis)
    n = int((l + 1) / 2)

    # # [X,ei] = e_{i+1} mod (X)
    for i in range(4, l - 2):
        for j in range(4, i + 1):
            s[K[3, i, j]] = 0

    # # [e1,e8]=-N
    for i in range(3, l - 1):
        s[K[4, l - 2, i]] = 0

    V_ind = [list(range(4, n))]
    for i in range(1, n - 3):
        V_ind.append(list(range(4, n - i + 1)))

    J_ind = []
    for i in range(0, n - 3):
        J_ind.append(list(range(3, n + 1 + i)))

    for i in range(n - 3):
        # # [V_i,V_i] involutive
        for a in V_ind[i]:
            for b in V_ind[i]:
                for c in range(3, l):
                    if c not in V_ind[i]:
                        s[K[a, b, c]] = 0

            # # [V_i,J^i]<J^i
        for a in V_ind[i]:
            for b in J_ind[i]:
                for c in range(3, l):
                    if c not in J_ind[i]:
                        s[K[a, b, c]] = 0
    return Particular_distr(
        g,
        s,
        Jacobi_id_wght,
        unshelve,
        reshelve,
        name="Invol_Prenorm_distr" + str(len(g.basis)),
        constant_coeff=constant_coeff,
    )


def Invol_6_Prenorm_distr(
    g:SympSymb, Jacobi_id_wght:int=0, unshelve:bool=False, reshelve:bool=False, constant_coeff:bool=False
)->Tuple[Distr_of_constant_symbol,dict,list[sp.Expr]]:
    """Returns a symplectified (2,7) distr with normalization utilizing the
      involutivity conditions of symplectified distributions; that is...

    INPUTS:
    * 'g' - a symplectic Tanaka symbol"""

    K = IndexedBase("K")
    s = {}
    l = len(g.basis)
    n = int((l + 1) / 2)

    # # [X,ei] = e_{i+1} mod (X)
    for i in range(4, l - 2):
        for j in range(4, i + 1):
            s[K[3, i, j]] = 0

    # # [e1,e6]=-N
    for i in range(3, l - 1):
        s[K[4, l - 2, i]] = 0

    V_ind = [list(range(4, n))]
    for i in range(1, n - 3):
        V_ind.append(list(range(4, n - i + 1)))

    J_ind = []
    for i in range(0, n - 3):
        J_ind.append(list(range(3, n + 1 + i)))

    for i in range(n - 3):
        # # [V_i,V_i] involutive
        for a in V_ind[i]:
            for b in V_ind[i]:
                for c in range(3, l):
                    if c not in V_ind[i]:
                        s[K[a, b, c]] = 0

            # # [V_i,J^i]<J^i
        for a in V_ind[i]:
            for b in J_ind[i]:
                for c in range(3, l):
                    if c not in J_ind[i]:
                        s[K[a, b, c]] = 0
    return Particular_distr(
        g,
        s,
        Jacobi_id_wght,
        unshelve,
        reshelve,
        name="Invol_Prenorm_distr" + str(len(g.basis)),
        constant_coeff=constant_coeff,
    )


def Standard_Prenorm_distr(
    g:SympSymb, Jacobi_id_wght:int=0, unshelve:bool=False, reshelve:bool=False, constant_coeff:bool=False
)->Tuple[Distr_of_constant_symbol,dict,list[sp.Expr]]:
    """Returns a free distribution with a prenormalized frame; that is,
    [X,e_i] = e_{i+1}, [e_1,e_{2n-6}]=-N, and [X,e_{2n-6}] has neither e_{2n-7} nor e_{2n-6} componenent.
    INPUTS:
    * 'g' - a symplectic Tanaka symbol"""

    K = IndexedBase("K")
    s = {}
    l = len(g.basis)
    for k1 in range(4, l - 2):
        for k2 in range(3, k1 + 1):
            s[K[3, k1, k2]] = 0
    for k in range(3, l - 1):
        s[K[4, l - 2, k]] = 0

    # Canonical section
    s[K[3, l - 2, l - 2]] = 0
    # Projective parametrization
    s[K[3, l - 2, l - 3]] = 0
    return Particular_distr(
        g,
        s,
        Jacobi_id_wght,
        unshelve,
        reshelve,
        name="Standard_Prenorm_distr" + str(len(g.basis)),
        constant_coeff=constant_coeff,
    )


def Zero_Wilc_distr(
    g:SympSymb, Jacobi_id_wght:int=0, unshelve:bool=False, reshelve:bool=False, constant_coeff:bool=False
)->Tuple[Distr_of_constant_symbol,dict,list[sp.Expr]]:
    """Returns a free distribution with a prenormalized frame and zero Wilczynski invariants; that is,
    [X,e_i] = e_{i+1}, [e_1,e_{2n-6}] = -N, and [X,e_{2n-6}] has neither e_{2n-7} nor e_{2n-6} componenent,
    and K[3,9,6] = K[3,9,4] = 0.
    INPUTS:
    * 'g' - a symplectic(7) symbol"""

    K = IndexedBase("K")
    s = {}
    l = len(g.basis)
    for k1 in range(4, l - 2):
        for k2 in range(3, k1 + 1):
            s[K[3, k1, k2]] = 0
    for k in range(3, l - 1):
        s[K[4, l - 2, k]] = 0

    # Canonical section
    s[K[3, l - 2, l - 2]] = 0
    # Projective parametrization
    s[K[3, l - 2, l - 3]] = 0

    # Wilcinski invariants
    s[K[3, 9, 6]] = 0
    s[K[3, 9, 4]] = 0
    return Particular_distr(
        g,
        s,
        Jacobi_id_wght,
        unshelve,
        reshelve,
        name="Standard_Prenorm_distr" + str(len(g.basis)),
        constant_coeff=constant_coeff,
    )


def Weak_Prenorm_distr(
    g:SympSymb, Jacobi_id_wght:int=0, unshelve:bool=False, reshelve:bool=False, constant_coeff:bool=False
)->Tuple[Distr_of_constant_symbol,dict,list[sp.Expr]]:
    """Returns a free distribution with a prenormalized frame; that is,
    [X,e_i] = e_{i+1}, [e_1,e_{2n-6}]=-N, and [X,e_{2n-6}] has no e_{2n-6} componenent.
    INPUTS:
    * 'g' - a symplectic Tanaka symbol"""

    K = IndexedBase("K")
    s = {}
    l = len(g.basis)

    # # [X,ei]=e_{i+1}
    for k1 in range(4, l - 2):
        for k2 in range(3, k1 + 1):
            s[K[3, k1, k2]] = 0

    # # [e1,e{2n-6}]=-N
    for k in range(3, l - 1):
        s[K[4, l - 2, k]] = 0

    # Canonical section
    s[K[3, l - 2, l - 2]] = 0
    return Particular_distr(
        g,
        s,
        Jacobi_id_wght,
        unshelve,
        reshelve,
        name="Weak_Prenorm_distr" + str(len(g.basis)),
        constant_coeff=constant_coeff,
    )


def Weakest_Prenorm_distr(
    g:SympSymb, Jacobi_id_wght:int=0, unshelve:bool=False, reshelve:bool=False, constant_coeff:bool=False
)->Tuple[Distr_of_constant_symbol,dict,list[sp.Expr]]:
    """Returns a free distribution with a prenormalized frame; that is,
    [X,e_i] = e_{i+1} and [e_1,e_{2n-6}]=-N.
    INPUTS:
    * 'g' - a symplectic Tanaka symbol"""

    K = IndexedBase("K")
    s = {}
    l = len(g.basis)
    for k1 in range(4, l - 2):
        for k2 in range(3, k1 + 1):
            s[K[3, k1, k2]] = 0
    for k in range(3, l - 1):
        s[K[4, l - 2, k]] = 0

    return Particular_distr(
        g,
        s,
        Jacobi_id_wght,
        unshelve,
        reshelve,
        name="Weakest_Prenorm_distr" + str(len(g.basis)),
        constant_coeff=constant_coeff,
    )


def Flat_distr(g:TSymb, Jacobi_id_wght:int=0)->Tuple[Distr_of_constant_symbol,dict,list[sp.Expr]]:
    """Returns a flat distribution with for the given Tanaka symbol.
    INPUTS:
    * 'g' - a symplectic Tanaka symbol"""

    K = IndexedBase("K")
    s = {}
    l = len(g.basis)
    p = g.basis.index(g.m_basis[0])
    for k1 in range(p, l):
        for k2 in range(k1, l):
            for k3 in range(l):
                s[K[k1, k2, k3]] = 0
    return Particular_distr(g, s, Jacobi_id_wght)


def Free_distr(g:TSymb, Jacobi_id_wght:int=0)->Tuple[Distr_of_constant_symbol,dict,list[sp.Expr]]:
    """Returns a free distribution with for the given Tanaka symbol.
    INPUTS:
    * 'g' - a symplectic Tanaka symbol"""

    return Particular_distr(g, {}, Jacobi_id_wght)
