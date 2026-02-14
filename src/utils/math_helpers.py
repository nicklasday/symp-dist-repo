import sympy as sp
from typing import TYPE_CHECKING

__all__ = [
    "find_linear_terms",
    "find_a_linear_term",
    "wght_of_ind",
    "Indexed_obj_in_expr",
    "hrs_min_sec",
    "sort_basis_tuple",
    "permutation_sign",
    "str_from_vec",
    "tuple_to_str",
    "append_monomial",
    "mul_str",
    "perm_sign",
    "mut_mat_copy",
    "Indexed_factors",
    "wghted_mul",
]

if TYPE_CHECKING:
    from ..algebra.tanaka_symbols import TSymb


def find_linear_terms(q:sp.Expr)->set[sp.Indexed]:
    """Finds the set of linear terms (i.e., variables which can be
    isolated by linear operations) without coefficients in the polynomial p.
    INPUTS:
    * 'q' - a polynomial in symbols and sp.Indexed objects"""
    p = sp.expand(q)
    if isinstance(p,sp.Add):
        r:set[sp.Indexed] = set()
        for monom in p.as_coeff_add()[1]:
            r = r.union(find_linear_terms(monom))
        return r
    if isinstance(p,sp.Mul):
        if len(p.as_coeff_mul()[1]) > 1:
            return set()
        r = find_linear_terms(p.as_coeff_mul()[1][0])
        return r
    if isinstance(p,sp.Pow):
        if p.as_base_exp()[1] == 1:
            return {p}
        return set()
    if isinstance(p,sp.Symbol) or isinstance(p,sp.Indexed):
        return {p}
    return set()


def find_a_linear_term(q:sp.Expr, excl_list:list[sp.Indexed]=[])->sp.Indexed:
    """Finds a linear term (i.e., a variable which can be
    isolated by linear operations) in the polynomial p.
    INPUTS:
    * 'q' - a polynomial in symbols and sp.Indexed objects"""
    p = sp.expand(q)
    if type(p) == sp.Add:
        for monom in p.as_coeff_add()[1]:
            temp = find_a_linear_term(monom, excl_list)
            if temp is not None:
                return temp
        return None
    if type(p) == sp.Mul:
        if len(p.as_coeff_mul()[1]) > 1:
            return None
        temp = find_a_linear_term(p.as_coeff_mul()[1][0], excl_list)
        if temp is not None:
            return temp
    if type(p) == sp.Pow:
        if p.as_base_exp()[1] == 1:
            if p.base[p.indices[0:3]] not in excl_list:
                return p
            return find_a_linear_term(q.xreplace({p: 0}), excl_list)
        return None
    if type(p) in [sp.Symbol, sp.Indexed]:
        if p.base[p.indices[0:3]] not in excl_list:
            return p
        return find_a_linear_term(q.xreplace({p: 0}), excl_list)
    return None


def wght_of_ind(ind_obj:sp.Indexed, alg:'TSymb'):
    """Returns the weight of ind_obj with respect to the basis of alg
    Note: this is negative the weight of the corresponding tensor"""
    r = sum([alg.basis[i].wght for i in ind_obj.indices])
    return r - 2 * alg.basis[ind_obj.indices[2]].wght


def Indexed_obj_in_expr(expr:sp.Expr)->set[sp.Indexed]:
    from ..algebra.tensor_algebras import TensorAlgElt

    """Returns the set of sp.Indexed obejcts which appear in expr
    INPUTS:
    * 'expr' - an algebraic expression in sp.Indexed objects and symbols"""
    if isinstance(expr, sp.Pow):
        return Indexed_obj_in_expr(expr.as_base_exp()[0])
    if isinstance(expr, sp.Mul):
        return set.union(*[Indexed_obj_in_expr(A) for A in expr.as_coeff_mul()[1]])
    if isinstance(expr, sp.Add):
        return set.union(*[Indexed_obj_in_expr(A) for A in expr.as_coeff_add()[1]])
    if isinstance(expr, sp.Indexed):
        return set([expr])
    if isinstance(expr, TensorAlgElt):
        r:set[sp.Indexed] = set()
        for d in expr.vd:
            for w in expr.vd[d]:
                r = r.union(Indexed_obj_in_expr(expr.vd[d][w]))
        return r
    try:
        iter(expr)
        r = set()
        for a in expr:
            r = r.union(Indexed_obj_in_expr(a))
        return r
    except TypeError:
        pass
    return set()

def symb_in_expr(expr:sp.Expr)->set[sp.Symbol]:
    from ..algebra.tensor_algebras import TensorAlgElt

    """Returns the set of sp.Indexed obejcts which appear in expr
    INPUTS:
    * 'expr' - an algebraic expression in sp.Indexed objects and symbols"""
    if isinstance(expr, sp.Pow):
        return symb_in_expr(expr.as_base_exp()[0])
    if isinstance(expr, sp.Mul):
        return set.union(*[symb_in_expr(A) for A in expr.as_coeff_mul()[1]])
    if isinstance(expr, sp.Add):
        return set.union(*[symb_in_expr(A) for A in expr.as_coeff_add()[1]])
    if isinstance(expr, sp.Indexed):
        return set()
    if isinstance(expr, sp.Symbol):
        return set([expr])
    if isinstance(expr, TensorAlgElt):
        r:set[sp.Indexed] = set()
        for d in expr.vd:
            for w in expr.vd[d]:
                r = r.union(symb_in_expr(expr.vd[d][w]))
        return r
    try:
        iter(expr)
        r = set()
        for a in expr:
            r = r.union(symb_in_expr(a))
        return r
    except TypeError:
        pass
    return set()


def hrs_min_sec(sec_val)->str:
    hours = str(round(sec_val // (60**2)))
    minutes = str(round((sec_val // 60) % 60))
    seconds = str(round(sec_val % 60, 0))
    if sec_val // (60**2) != 0:
        return hours + " hrs " + minutes + " min " + seconds + " sec"
    if (sec_val // 60) % 60 != 0:
        return minutes + " min " + seconds + " sec"
    return seconds + " sec"


def sort_basis_tuple(basis_tuple, basis):
    """Returns: a tuple containing an rearrangement of basis_tuple of descending degree,
    and the sign of the permutation (either -1 or 1)

    INPUTS:
    * basis: a list of elements
    * basis_tuple: a tuple or list of elts from basis"""

    basis_list = list(basis_tuple)
    sorted_list = basis_list.copy()
    sorted_list.sort(key=lambda A: basis.index(A))
    return (tuple(sorted_list), permutation_sign(basis_list, sorted_list))


def permutation_sign(it_1, it_2):
    """tuple_1, tuple_2: iterables containing the same elements
    returns: the sign of the permutation taking it_1 to it_2"""
    cnt = 0
    for i in range(len(it_1)):
        for j in range(i + 1, len(it_1)):
            if it_2.index(it_1[j]) < it_2.index(it_1[i]):
                cnt += 1
    return (-1) ** cnt


# string manipulation/printing


def str_from_vec(v, b):
    r = []
    t = ""
    for i in range(len(v)):
        if v[i] != 0:
            r.append([v[i], b[i]])
    if len(r) == 0:
        return "0"
    for i in range(len(r)):
        if r[i][0] == -1:
            t = t + " - " + str(r[i][1])
        elif r[i][0] != 1:
            if type(r[i][0]) == sp.Add:
                str_a = "(" + str(r[i][0]) + ")"
            else:
                str_a = str(r[i][0])
            if i == 0:
                t = t + str_a + "*" + str(r[i][1])
            else:
                t = t + " + " + str_a + "*" + str(r[i][1])
        else:
            if i == 0:
                t = str(r[i][1])
            else:
                t = t + " + " + str(r[i][1])
    return t


def tuple_to_str(v):
    if v == tuple():
        return str(v)
    r = "("
    for A in v:
        r = r + str(A) + ","
    r = r[0:-1] + ")"
    return r


def append_monomial(s, c, p):
    """Returns a new string consisting of s + c*p
    INPUTS:
    * 's' - a string
    * 'c' - a (numeric or symbolic) coefficient
    * 'p' - a string or basis element
    """
    m = mul_str(c, p)
    if m == "":
        return s
    if s == "":
        return m
    if m[0] in ["-", "+"]:
        return s + m
    return s + "+" + m


def mul_str(c, p):
    """Returns a string consisting of c*p
    INPUTS:
    * 'c' - a (numeric or symbolic) coefficient
    * 'p' - a string or basis element
    """
    if c == 0:
        return ""
    if c == -1:
        return "-" + str(p)
    if c == 1:
        return str(p)
    if type(c) == sp.Add:
        return "(" + str(c) + ")*" + str(p)
    return str(c) + "*" + str(p)


def perm_sign(L, sort_L):
    """args: two lists with the same elements;
    returns: the sign of the permutation (slow)"""
    result = 1
    for j in range(len(L)):
        pj = sort_L.index(L[j])
        for i in range(j):
            if pj < sort_L.index(L[i]):
                result = result * (-1)
    return result


def mut_mat_copy(A):
    r = sp.zeros(*sp.shape(A))
    for i in range(sp.shape(A)[0]):
        for j in range(sp.shape(A)[1]):
            r[i, j] = A[i, j]
    return r


def Indexed_factors(expr):
    """args: expr, a polynomial in sp.Indexed objects
    returns: a set of the sp.Indexed objects involved in expr"""
    if type(expr) == sp.Pow:
        return sp.Indexed_factors(expr.as_base_exp()[0])
    if type(expr) == sp.Mul:
        return set.union(*[sp.Indexed_factors(A) for A in expr.as_coeff_mul()[1]])
    if type(expr) == sp.Add:
        return set.union(*[sp.Indexed_factors(A) for A in expr.as_coeff_add()[1]])
    if type(expr) == sp.Indexed:
        return set([expr])
    return set()


def wghted_mul(M1, w1, M2, w2, m):
    """args: M and N, a (2m+5)x(2m+5) matrix of degree w1 and w2, respectively
    returns: MN"""
    f = [0, 1, 3] + list(range(4, 2 * m + 6))  # f[i]:f[i+1] gives the ith piece
    r = sp.zeros(2 * m + 5)
    if w1 + w2 > 2 * m + 4:
        return r
    for i in range(2 * m + 4 - w1 - w2):
        j = i + w1 + w2
        M1_s = M1[f[i] : f[i + 1], f[i + w1] : f[i + w1 + 1]]
        M2_s = M2[f[i + w1] : f[i + w1 + 1], f[j] : f[j + 1]]
        r[f[i] : f[i + 1], f[j] : f[j + 1]] = M1_s * M2_s
    return r
