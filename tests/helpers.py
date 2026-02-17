from symp_dist.algebra.complexes.cochain import Cochain
from symp_dist.algebra.tanaka_symbols.tanaka_symbol_basis_elt import TSymbBasisElt
from symp_dist.algebra.tanaka_symbols.tanaka_symbol_elt import TSymbElt
import sympy as sp
import symp_dist.utils.distribution_helpers as dh


def SF_ad(X1, X2, SF:Cochain):
    """args: X1,X2 are VFs, either T_elts or vectors/lists with coeffs indexed objects and
          the coordinates y,h,e SF is the structure function defining the ad-relations between
          T basis elts, an element of C, the cochain complex
    returns: [X1,X2], where the str function defines the relations between T basis elts,
            and K,y,h,e depend on the coordinates appropriately"""
    # from tanaka_symbol_elt import TSymbElt
    # from .tanaka_symbol_basis_elt import TSymbBasisElt

    T = SF.parent.alg
    result = 0
    X:list = [X1, X2]
    v:list = [None, None]
    for j in range(2):
        Xj = X[j]
        v[j]
        if type(Xj) == list:
            v[j] = Xj
        if type(Xj) == type(sp.eye(4)):
            v[j] = list(Xj)
        if type(Xj) == TSymbElt or type(Xj) == TSymbBasisElt:
            v[j] = Xj.vec_rep
    for i in range(len(v[0])):
        coeff_1 = v[0][i]
        if coeff_1 != 0:
            for j in range(len(v[1])):
                coeff_2 = v[1][j]
                if coeff_2 != 0:
                    res1 = coeff_1 * dh.abn_ind_der(coeff_2, i) * T.basis[j]
                    res2 = -coeff_2 * dh.abn_ind_der(coeff_1, j) * T.basis[i]
                    result += res1
                    result += res2
    e_elt = T.elt(v[0]).cast_as_ext_elt().wedge(T.elt(v[1]).cast_as_ext_elt())
    result += SF.apply_cochain_map(e_elt)
    return result
