__all__ = ["dds_back_substitute", "dds_subs", "dds_val", "add_DiffOp_to_dds_dict"]

import sympy as sp
from ..algebra.complexes import cochain
from .diff_op_dict import DiffOpDict
from ..utils import math_helpers as mh
import copy

# These methods are only used for applying the Ricci identity,
# which I think turned out to be unnecessary for computing the syzygies.
# Maybe I'll circle back to this if needed and make dds_dict (diff diff subs dict)
# into a class of its own. For the moment, I'll leave it as is.


def dds_back_substitute(d1, d2, symb, curv):
    """Back substitutes d2 into d1
    INPUTS:
    * 'd1,d2' - diff diff subs dictionaries
    * 'symb' - a Tanaka symbol
    * 'curv' - a curvature 2-cochain"""
    # substitute within keys of d1
    for p in d2:
        for k1 in list(d1.keys()):
            if k1 in d1:  # the keys of d1 might be modified by add_DiffOp_to_dds_dict
                i = bytes(k1).find(bytes(p))
                if i != -1:
                    new_op = DiffOpDict(d1[k1], symb, curv) - DiffOpDict(
                        {k1[0:i]: 1}, symb, curv
                    ) * DiffOpDict(d2[p], symb, curv) * DiffOpDict(
                        {k1[i + len(p) : len(k1)]: 1}, symb, curv
                    )
                    d1.pop(k1)
                    add_DiffOp_to_dds_dict(new_op, d1, symb, curv)

    # Check if more substitutions are needed in the keys
    first_subs_complete = True
    for p in d2:
        for k1 in d1:
            i = bytes(k1).find(bytes(p))
            if i != -1:
                first_subs_complete = False
    if not first_subs_complete:
        return dds_back_substitute(d1, d2, symb, curv)

    # Sub into the values
    for k in d1:
        d1[k] = dds_subs(d1[k], d2, symb, curv)


def dds_subs(expr, dds_dict, symb, curv):
    if type(expr) == cochain:
        r = expr.parent.elt({})
        for a in expr.vd:
            if a not in r.vd:
                r.vd[a] = {}
            for b in expr.vd[a]:
                r.vd[a][b] = dds_subs(expr.vd[a][b], dds_dict, symb, curv)
        return r

    if type(expr) in [sp.Matrix, sp.MutableDenseMatrix, sp.ImmutableDenseMatrix]:
        r = sp.Matrix(expr)
        for i in range(sp.shape(r)[0]):
            for j in range(sp.shape(r)[1]):
                r[i, j] = dds_subs(r[i, j], dds_dict, symb, curv)
        return r

    if type(expr) == tuple:  # this represents a key from a DiffOpDict
        for p in dds_dict:
            i = bytes(expr).find(bytes(p))
            if i != -1:
                r = (
                    DiffOpDict({expr[0:i]: 1}, symb, curv)
                    * DiffOpDict(dds_dict[p], symb, curv)
                    * DiffOpDict({expr[i + len(p) : len(expr)]: 1}, symb, curv)
                )
                return dds_subs(r, dds_dict, symb, curv)
        return DiffOpDict({expr: 1}, symb, curv)

    if type(expr) == DiffOpDict:
        r = DiffOpDict({}, symb, curv)
        for k in expr.d:
            r += dds_subs(expr.d[k], dds_dict, symb, curv) * dds_subs(
                k, dds_dict, symb, curv
            )
        return r
    if not hasattr(expr, "subs"):
        return expr

    I_set = mh.Indexed_obj_in_expr(expr)
    nd = {}
    for I in I_set:
        v = dds_val(dds_dict, I, symb, curv)
        if v != I:
            nd[I] = v
    r = expr.xreplace(nd)
    return r


def dds_val(dds_dict, a, symb, curv):
    """returns the value of an indexed object after applying substitutions from dds_dict
    ARGS:
    * 'ind_obj' - an indexed object with base among I, W, K, eta, and alpha
    * 'dds_dict' - a differential differential substution dictionary"""
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
        print("WARNING: dds_subs expects only bases I, W, K, eta, and alpha")

    # Pick a key which appears and substitute
    # Careful! Indices are reversed in the operators as compared to the tensors
    k = None
    i = None
    for k1 in dds_dict:
        inds = k1[::-1]
        i1 = bytes(a.indices[basic_len : len(a.indices)]).find(bytes(inds))
        if i1 != -1:
            k, i = k1, i1 + basic_len
            break
    if i is None:
        return a
    r = a.base[a.indices[0:i]]
    inds = a.indices[i + len(k) : len(a.indices)][::-1]
    postf = DiffOpDict({inds: 1}, symb, curv)
    curr = DiffOpDict(dds_dict[k], symb, curv)
    r = postf.apply(curr.apply(r))
    # Finally, substitute the rest
    return dds_subs(r, dds_dict, symb, curv)


def add_DiffOp_to_dds_dict(DiffOp, dds_dict, symb, curv):
    """Adds the relation DiffOp=0 to the given dds_dict"""
    DiffOp.clear_zeros()
    DiffOp = DiffOp.normal_form()
    # for k in dds_dict:
    #     for j in dds_dict[k]:
    #         dds_dict[k][j]=dds_dict[k][j] # This is inefficient, but I need to make sure the values are nonzero somehow
    # Pick a key to substitute for
    # Choose the key of greatest length which comes first in revlex ordering
    key_len = -1
    for k in DiffOp.d:
        if len(k) > key_len:
            key_len = len(k)
    if key_len == -1:
        return None
    key_list = sorted([k for k in DiffOp.d if len(k) == key_len])
    key_list.reverse()
    iso_key = None
    for k in key_list:
        if type(sp.simplify(DiffOp.d[k])) == int:
            if sum([symb.basis[i].wght >= 0 for i in k]) == 0:
                iso_key = k
                break
    if iso_key is None:
        iso_key = key_list[0]
    DiffOp.d[iso_key] = sp.simplify(DiffOp.d[iso_key])
    if DiffOp.d[iso_key] == 0:
        DiffOp.d.pop(iso_key)
        add_DiffOp_to_dds_dict(DiffOp, dds_dict, symb, curv)
        return None
    try:
        new_val = copy.deepcopy((-sp.Rational(1, DiffOp.d[iso_key]) * DiffOp).d)
    except:
        new_val = copy.deepcopy(((-1 / DiffOp.d[iso_key]) * DiffOp).d)
    new_val.pop(iso_key)
    for k in new_val:
        new_val[k] = sp.simplify(new_val[k])
    new_dds = {iso_key: new_val}
    dds_back_substitute(dds_dict, new_dds, symb, curv)
    dds_dict[iso_key] = new_val
