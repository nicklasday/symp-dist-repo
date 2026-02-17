
from __future__ import annotations
from typing import TYPE_CHECKING

import sympy as sp
from ..distributions.diff_op_dict import DiffOpDict
from ..algebra.complexes import cochain
from ..utils import math_helpers as mh
import copy


if TYPE_CHECKING:
    from ..algebra.complexes.cochain import Cochain
    from ..distributions import Distr_of_constant_symbol
    from ..algebra.tanaka_symbols import TSymb

def ds_val(ds_dict:dict[sp.Indexed,dict[tuple[int,...],sp.Expr]], a:sp.Indexed)->tuple[sp.Expr,set[tuple[sp.Indexed,tuple[int,...]]]]:
    """Returns the value of 'a' according to ds_dict along with the keys of ds_dict used in the substitution
    INPUTS:
    * 'ds_dict' - a differential substitution dictionary
    * 'a' - an Indexed object, representing a derivative of a 
            (2,1)-tensor with base K, alpha, or eta; or an invariant I or W
    """
    from ..distributions import Distr_of_constant_symbol


    basic_len = -1
    if a.base in [sp.IndexedBase("I"), sp.IndexedBase("W")]:
        basic_len = 1
    elif a.base in [
        sp.IndexedBase("K"),
        sp.IndexedBase("alpha"),
        sp.IndexedBase("eta"),
    ]:
        basic_len = 3
    else:
        print("WARNING: ds_val expects only bases I, W, K, eta, and alpha")

    # Speed this up! I think there should be a way.
    if a.base[a.indices[0:basic_len]] not in ds_dict:
        return (a, set())
    pfs = [
        [a.indices[basic_len:l], a.indices[l : len(a.indices)]]
        for l in range(basic_len, len(a.indices) + 1)
    ]
    pfs.reverse()
    for pf in pfs:
        if pf[0] in ds_dict[a.base[a.indices[0:basic_len]]]:
            r = ds_dict[a.base[a.indices[0:basic_len]]][pf[0]]
            for i in pf[1]:
                r = Distr_of_constant_symbol.abn_ind_der(
                    r, i
                )  # Should I be caching this? It might become very large
            result, rel_keys = ds_subs(r, ds_dict)
            return (result, rel_keys.union({(a.base[a.indices[0:basic_len]], pf[0])}))
    return (a, set())

def add_expr_to_ds_dict(
    expr:sp.Expr,
    ds_dict:dict[sp.Indexed,dict[tuple[int,...],sp.Expr]],
    g:TSymb,
    rel_Bianchi_terms:set|None=None,
    fund_invars:list[sp.Indexed]=[],
    rel_Bianchi_dict:dict|None=None,
    not_added:list[sp.Expr]|None=None,
)->None:
    """ "Computes a substitution from the equation expr==0, substituting back into ds_dict.
    Assumes expr has already been substituted with ds_dict.
    Used, for example, to reprocess a key like K[3,4,5,3] when K[3,4,5] is processed."""
    # expr may already be in the ideal ds_dict
    if expr == 0:
        return None

    # There may be no linear monomials to isolate without branching
    curr_K = mh.find_a_linear_term(expr, fund_invars)
    if curr_K is None:
        curr_K = mh.find_a_linear_term(expr)
    if curr_K is None:
        if not_added is not None:
            not_added.append(expr)
        if rel_Bianchi_dict is not None:
            rel_Bianchi_dict[expr] = rel_Bianchi_terms
        return None

    # list_to_avoid=[]
    # # # (2,6) case
    # if len(Distr.basis)==8: list_to_avoid=[(3,9,6),(3,9,4),(3,9,8),(3,9,9),(4,8,6),(4,8,7),(4,7,5)]
    # # # (2,7) case
    # if len(Distr.basis)==10: list_to_avoid=[(3,11,8),(3,11,6),(3,11,4),(3,11,11),(3,11,10)]

    # # # If the only linear terms are from list_to_avoid this will choose one
    # if curr_K.indices[0:3] in list_to_avoid:
    #     F=list(find_linear_terms(expr))
    #     # We'll try to avoid using the Wilcinski invariants whenever possible
    #     i=0
    #     j=0
    #     while i<len(F):
    #         if F[i].indices[0:3] in list_to_avoid: i+=1
    #         else:
    #             j=i
    #             i=len(F)
    #     curr_K=F[j]

    # Solve; add to ds_dict
    try:
        s = sp.solve(expr, curr_K)[0]
    except:
        temp = {Kijk: sp.Symbol(str(Kijk)) for Kijk in mh.Indexed_obj_in_expr(expr)}
        temp_inv = {temp[Kijk]: Kijk for Kijk in temp}
        s = sp.solve(expr.xreplace(temp), sp.Symbol(str(curr_K)))[0].xreplace(temp_inv)
    ds_add_key(
        curr_K, s, ds_dict, g, rel_Bianchi_terms, rel_Bianchi_dict, not_added
    )


def ds_subs(expr:sp.Expr, 
            ds_dict:dict[sp.Indexed,dict[tuple[int,...],sp.Expr]], 
            )->tuple[sp.Expr,set[tuple[sp.Indexed,tuple[int,...]]]]:
    """Applies the substitutions from ds_dict to expr, returning the substituted expression
    and the set of keys from ds_dict used in the subsitution.
    INPUTS:
    * 'expr' - a cochain or expression
    * 'ds_dict' - a differential substitution dictionary
    * 'Distr' - a distribution of constant symbol"""
    keys_used:set[tuple[sp.Indexed,tuple[int,...]]] = set()
    if type(expr) == cochain:
        r = expr.parent.elt({})
        for a in expr.vd:
            if a not in r.vd:
                r.vd[a] = {}
            for b in expr.vd[a]:
                r.vd[a][b], temp_keys_used = ds_subs(expr.vd[a][b], ds_dict)
                keys_used = keys_used.union(temp_keys_used)
        return (r, keys_used)

    if type(expr) in [sp.Matrix, sp.MutableDenseMatrix, sp.ImmutableDenseMatrix]:
        r = sp.Matrix(expr)
        for i in range(sp.shape(r)[0]):
            for j in range(sp.shape(r)[1]):
                r[i, j], temp_keys_used = ds_subs(r[i, j], ds_dict)
                keys_used = keys_used.union(temp_keys_used)
        return (r, keys_used)

    if type(expr) == DiffOpDict:
        r = copy.deepcopy(expr)
        for k in r.d:
            r.d[k], temp_keys_used = ds_subs(r.d[k], ds_dict)
            keys_used = keys_used.union(temp_keys_used)
        r.clear_zeros()
        return (r, keys_used)

    if not hasattr(expr, "subs"):
        return (expr, keys_used)
    I_set = mh.Indexed_obj_in_expr(expr)
    nd = {}
    for I in I_set:
        v, temp_keys_used = ds_val(ds_dict, I)
        if v != I:
            nd[I] = v
            keys_used = keys_used.union(temp_keys_used)
    r = expr.xreplace(nd)
    return (r, keys_used)


def ds_back_substitute(d1:dict[sp.Indexed,dict[tuple[int,...],sp.Expr]], 
                       d2:dict[sp.Indexed,dict[tuple[int,...],sp.Expr]], 
                       g:TSymb, 
                       min_wght:int, 
                       rel_Bianchi_dict=None, 
                       not_added=None)->None:
    """Back substitutes d2 into d1
    INPUTS:
    * 'd1,d2' - differential substitution dictionaries
    * 'g' - a TSymb elt, whose weights will be used in calculations
    * 'min_wght' - the minimal weight of the keys of d2"""
    temp_rel_Bianchi_dict = {}
    removed_ids = {}
    for k2 in d2:
        if k2 in list(d1.keys()):
            for i2 in d2[k2]:
                for i1 in list(d1[k2].keys()):
                    if i1[0 : len(i2)] == i2 and len(i2) < len(i1):
                        r_Id = k2.base[(*k2.indices, *i1)] - d1[k2][i1]
                        removed_ids[(k2, i1)] = r_Id
                        if rel_Bianchi_dict is not None:
                            # Move the relevant key list over
                            temp_rel_Bianchi_dict[r_Id] = copy.copy(
                                rel_Bianchi_dict[k2][i1]
                            )
                            del rel_Bianchi_dict[k2][i1]
                        del d1[k2][i1]
    # Back substitute
    for a in d1:
        for b in d1[a]:
            my_wght = mh.wght_of_ind(a, g)
            for i in b:
                my_wght += -g.basis[i].wght
            if my_wght >= min_wght:
                d1[a][b], temp_keys = ds_subs(d1[a][b], d2)
                if rel_Bianchi_dict is not None:
                    temp_rel_Bianchi_keys = set().union(
                        *[rel_Bianchi_dict[c[0]][c[1]] for c in temp_keys]
                    )
                    if a not in rel_Bianchi_dict: rel_Bianchi_dict[a]={}
                    rel_Bianchi_dict[a][b] = rel_Bianchi_dict[a][b].union(
                        temp_rel_Bianchi_keys
                    )
    # Add back the removed identities
    for rI_key in removed_ids:
        rI = removed_ids[rI_key]
        rI2, temp_keys2 = ds_subs(rI, d2)
        rI1, temp_keys1 = ds_subs(rI2, d1)
        if rel_Bianchi_dict is not None:
            temp_rel_Bianchi_keys = temp_rel_Bianchi_dict[rI]
            # print('removed_ids :',removed_ids)
            for a in temp_keys1.union(temp_keys2):
                # print('a =',a)
                try:
                    temp_rel_Bianchi_keys = temp_rel_Bianchi_keys.union(
                        rel_Bianchi_dict[a[0]][a[1]]
                    )
                except:
                    temp_rel_Bianchi_keys = temp_rel_Bianchi_keys.union(
                        temp_rel_Bianchi_dict[removed_ids[a]]
                    )
            add_expr_to_ds_dict(
                rI1, d1, g, temp_rel_Bianchi_keys, rel_Bianchi_dict, not_added
            )  # This is changing keys of d1 and rel_Bianchi_dict
        else:
            add_expr_to_ds_dict(rI, d1, g)
    return None


def ds_add_key(
    key:sp.Indexed,
    val:sp.Expr,
    ds_dict:dict[sp.Indexed,dict[tuple[int,...],sp.Expr]],
    g:TSymb,
    rel_Bianchi_keys=None,
    rel_Bianchi_dict=None,
    not_added=None,
):
    """adds the key, value pair to the differential substitution dictionary ds_dict,
    back substituting if necessary. Overwrites existing key if already present
    ARGUMENTS:
    * 'key' - an indexed object
    * 'val' - the value the indexed object should be replaced with
    * 'ds_dict' - the ds_dict to which the key-value pair should be added
    * 'g' - a TSymb elt, whose weights will be used in calculations
    * 'rel_Bianchi_keys' - a set of tuples referring to components of the Bianchi identity
    * 'rel_Bianchi_dict' - a dictionary keeping track of the components of the Bianchi identity
                           being which are needed to derive a given relation
    """
    b_key = key.base[key.indices[0:3]]
    post = key.indices[3 : len(key.indices)]
    if rel_Bianchi_keys is not None:
        if b_key not in rel_Bianchi_dict:
            rel_Bianchi_dict[b_key] = {}
        rel_Bianchi_dict[b_key][post] = rel_Bianchi_keys
    # first back substitute
    ds_back_substitute(
        ds_dict, {b_key: {post: val}}, g, -sp.oo, rel_Bianchi_dict, not_added
    )
    # then add the key if it's not redundant
    new_expr, rel_keys = ds_subs(key - val, ds_dict)
    if rel_Bianchi_dict is not None:
        temp_rel_Bianchi_keys = set().union(
            *[rel_Bianchi_dict[a[0]][a[1]] for a in rel_keys]
        )
    else:
        temp_rel_Bianchi_keys = None
    if new_expr != key - val:
        add_expr_to_ds_dict(new_expr, ds_dict, g, temp_rel_Bianchi_keys, not_added)
    else:
        if b_key not in ds_dict:
            ds_dict[b_key] = {}
        ds_dict[b_key][post] = val
        if rel_Bianchi_dict is not None:
            if b_key not in rel_Bianchi_dict:
                rel_Bianchi_dict[b_key] = {}
            if post not in rel_Bianchi_dict[b_key]:
                rel_Bianchi_dict[b_key][post] = set()
            rel_Bianchi_dict[b_key][post] = rel_Bianchi_dict[b_key][post].union(
                temp_rel_Bianchi_keys
            )


def subs_needed(expr:sp.Expr, ds_dict:dict[sp.Indexed,dict[tuple[int,...],sp.Expr]])->bool:
    I = mh.Indexed_obj_in_expr(expr)
    for A in I:
        eta = A.base
        pre = A.indices[0:3]
        post = A.indices[3 : len(A.indices)]
        if eta[pre] in ds_dict:
            for i in range(len(post) + 1):
                if post[0:i] in ds_dict[eta[pre]]:
                    return True
    return False


def ds_subs_needed(ds_dict:dict[sp.Indexed,dict[tuple[int,...],sp.Expr]])->bool:
    """Returns True if the values of the differential substitution dictionary ds_dict
        involve its keys, so that additional back substitution is possible. Returns False otherwise. """
    for k in ds_dict:
        for j in ds_dict[k]:
            if subs_needed(ds_dict[k][j], ds_dict):
                print(k, j)
                return True
    return False


def process_JI(
    ind:tuple[int,int,int,int], 
    Distr:Distr_of_constant_symbol, 
    ds_dict:dict[sp.Indexed,dict[tuple[int,...],sp.Expr]], 
    fund_invars:list[sp.Indexed]=[], 
    rel_Bianchi_dict=None, 
    not_added=None
)->None:
    """Adds the Jacobi identity of Distr corresponding to ind to
    the differential substitution dict ds_dict.

    If no linear monomial can be found, raises a no_linear_monomial_exception

    INPUTS:
    * 'ind' - a 4-tuple of natural numbers > 2
    * 'ds_dict' - a dict representing differential substitutions from a Jacobi ideal
    """
    expr, temp_keys = ds_subs(Distr.Jacobi_id(*ind), ds_dict)
    expr = sp.expand(expr)
    if rel_Bianchi_dict is not None:
        add_expr_to_ds_dict(
            expr,
            ds_dict,
            Distr.Tanaka_symbol,
            {ind}.union(temp_keys),
            fund_invars,
            rel_Bianchi_dict,
            not_added,
        )
    else:
        add_expr_to_ds_dict(
            expr, ds_dict, Distr.Tanaka_symbol, fund_invars=fund_invars, not_added=not_added
        )
    return None