from typing import TYPE_CHECKING

import sympy as sp
from .lin_alg_helpers import coords_to_lin_comb

if TYPE_CHECKING:
    from src.algebra.complexes import Cochain


def find_cochain_basis(ss:list['Cochain'])->list['Cochain']:
    """args: ss (spanning set), a list of cochains of the same homogeneous degree
    Returns: A list of cochains which are a basis for the subspace spanned by ss"""
    from src.algebra.complexes import Cochain
    if len(ss) == 0:
        return []
    basis_set = set()
    for c in ss:
        for base_elt in c.coeff_dict:
            basis_set.add(base_elt)
    B = list(basis_set)

    M = sp.zeros(len(ss), len(B))
    for i in range(len(ss)):
        sp.set_row(M, i, coordinatize_cochain_in_basis(ss[i], B))
    M = M.rref()[0]
    result = [
        list(M.row(i))
        for i in range(sp.shape(M)[0])
        if list(M.row(i)) != [0] * len(M.row(i))
    ]
    basis_cochains = [Cochain({A: 1}, ss[0].parent) for A in B]
    return [coords_to_lin_comb(A, basis_cochains) for A in result]


def simplify_cochain(c:'Cochain')->None:
    for d in c.vd:
        for w in c.vd[d]:
            c.vd[d][w] = sp.simplify(c.vd[d][w])
    c.clear_zeros()
    return None


def coordinatize_cochain_in_basis(c, basis):
    """c: a cochain object of homogeneous degree
    basis: a collection of str tuples representing elementary cochains
    returns the vector representation of c w.r.t basis as a list"""
    result = [0] * len(basis)
    for key in c.coeff_dict:
        if key in basis:
            result[basis.index(key)] = c.coeff_dict[key]
        else:
            print(
                "coordinatize_cochain_in_basis error: cochain component",
                key,
                "not in basis",
            )
    return result
