from itertools import combinations
from typing import Union, TYPE_CHECKING
import sympy as sp
import random


from .math_helpers import wghted_mul, append_monomial
from ..utils.exceptions import wedge_Mat_Exception
from ..utils.sympy_utils import to_sympy_matrix

if TYPE_CHECKING:
    from utils.types import VectorInput

__all__ = [
    "ind_diag",
    "colspace_containment",
    "Mat_adjoint",
    "ortho_proj",
    "update_add_mat",
    "nilp_exp",
    "set_row",
    "set_col",
    "wedge_vecs",
    "cd_to_vec",
    "remove_zeros_vd",
    "str_from_vd",
    "iprod_mat",
    "augment_with_identity",
    "Mat_preim_elt",
    "new_Mat_preim_elt",
    "ZV",
    "col_sp_and_preim",
    "wedge_Mat",
    "kUT_test",
    "homog_exp",
    "rand_UT",
    "coords_to_lin_comb",
]


def ind_diag(*elts):
    """
    this is a version of diag with accepts Indexed objects
    """
    r = sp.zeros(len(elts))
    for i in range(len(elts)):
        r[i, i] = elts[i]
    return r


def colspace_containment(M, N):
    """Checks if the columnspace of M is contained in the columnspace of N
    INPUTS:
    * 'M', 'N' - Matrices
    """
    N_cols = [list(N.col(i)) for i in N.rref()[1]]
    col_list = N_cols + [list(M.col(i)) for i in range(sp.shape(M)[1])]
    return len(sp.Matrix(col_list).rref()[1]) <= len(N_cols)


def Mat_adjoint(A, Q_dom, Q_codom):
    """Returns the adjoint matrix of A with respect to the inner products
    defined by Q_dom and Q_codom
    INPUTS:
    * 'A' - A matrix representing a linear transformation from dom-->codom
    * 'Q_dom' - A matrix representing a 2-form on dom
    * 'Q_codom' - A matrix representing a 2-form on codom
    """
    return (Q_codom * A * Q_dom.inv()).transpose()


def ortho_proj(v, basis, Q):
    """Gives the orthogonal projection with respect to Q of
    the vector v onto the space spanned by basis
    INPUTS:
    * 'v' - a list or vector
    * 'basis' - a matrix with linearly independent columns of length len(v)
    * 'Q' - a len(v)xlen(v) matrix, representing a 2-form
    """
    # Convert v and basis to SymPy matrices (column vectors)
    u = sp.SparseMatrix(v)
    if sp.shape(u)[1] != 1:
        u = u.transpose()

    A = sp.SparseMatrix(basis)
    if sp.shape(A)[1] == 0:
        return sp.zeros(*sp.shape(u))

    # Compute Gramian matrix G
    G = A.transpose() * Q * A

    # Solve linear system G * x = A.T * Q * v
    x = G.solve(A.transpose() * Q * u)

    # Compute projection
    projection = A * x

    return projection


def update_add_mat(Md, d, w, M):
    """Adds vector v to the specified degree and weight of self
    INPUTS:
    * 'Md' - Matrix dictionary
    * 'd' - degree
    * 'w' - weight
    * 'M' - Matrix
    """
    if d not in Md:
        Md[d] = {}
    if w not in Md[d]:
        Md[d][w] = sp.SparseMatrix(M)
    else:
        Md[d][w] = sp.SparseMatrix(Md[d][w] + M)


def update_add_dict(Md1, Md2, coeff=1):
    """Adds the vector dict coeff*Md2 to Md1, updating Md1
    INPUTS:
    * 'Md1','Md2' - matrix dictionaries
    * 'coeff' - a coefficient
    """
    if coeff == 0:
        return None
    for d in Md2:
        for w in Md2[d]:
            update_add_mat(Md1, d, w, coeff * Md2[d][w])


def nilp_exp(M, step=None):
    """Computes (exp(M),exp(-M)) for a nilpotent matrix M with M**step=0
    INPUTS:
    * 'M' -- a nilpotent matrix
    * 'step' -- an integer so that M**step=0
    """
    p_cache = [sp.eye(*sp.shape(M))]
    if step is not None:
        for i in range(1, step):
            p_cache.append(p_cache[-1] * M)
    else:
        M_to_i=p_cache[0]
        while M_to_i!=sp.zeros(*sp.shape(M)):
            p_cache.append(p_cache[-1] * M)
    r1 = sum([p_cache[i] / sp.factorial(i) for i in range(1, len(p_cache))], p_cache[0])
    r2 = sum(
        [(-1) ** i * p_cache[i] / sp.factorial(i) for i in range(1, len(p_cache))],
        p_cache[0],
    )
    return (r1, r2)


def set_row(mat, rowNum, row):
    if type(row) == type(sp.zeros(3, 3)):
        rowList = list(row)
    else:
        if type(row) == type([0]):
            rowList = row
        else:
            print("setRow error: arg row must be either matrix or list")
    if len(rowList) != len(mat.row(0)):
        print("setRow error: mat.row() and row have differing lengths")
        return None
    for i in range(len(rowList)):
        mat[rowNum, i] = rowList[i]


def set_col(mat, colNum, col):
    colList = list(col)
    if len(colList) != len(mat.col(0)):
        print("SetCol error: mat.col() and col have differing lengths")
        return None
    for i in range(len(colList)):
        mat[i, colNum] = colList[i]


def wedge_vecs(vec_list):
    """Returns a vector representing the wedge product of the given vectors"""
    if len(vec_list) == 0:
        return None
    M = sp.Matrix(list(map(list, vec_list)))
    result = sp.zeros(sp.binomial(len(vec_list[0]), len(vec_list)), 1)
    L = list(combinations(range(len(vec_list[0])), len(vec_list)))
    for j in range(len(L)):
        result[j] += sp.SparseMatrix([list(M.col(k)) for k in L[j]]).det()
    return result


def cd_to_vec(cd, base, base_index_dict):
    v = sp.SparseMatrix([0] * len(base))
    print(v)
    for k in cd:
        v[base_index_dict[k]] = cd[k]
    return v


def remove_zeros_vd(vd):
    # This could be quite costly if the equality checks are complicated...
    vdk = list(vd.keys())
    for d in vdk:
        vddk = list(vd[d].keys())
        for w in vddk:
            if vd[d][w] == sp.zeros(*sp.shape(vd[d][w])):
                vd[d].pop(w)
        if vd[d] == {}:
            vd.pop(d)


def str_from_vd(vd:dict[int,dict[int,sp.MatrixBase]], b)->str:
    """Produces a string from a vector dictionary
    INPUTS:
    * vd - a vector dictionary; a dict of dicts of vectors with keys deg then wght
    * b - a basis method returning a dict of dicts of basis elts, with keys deg then wght
    """
    r = ""
    for d in vd:
        for w in vd[d]:
            for i in range(len(vd[d][w])):
                r = append_monomial(r, vd[d][w][i], b(d, w)[i])
    if r == "":
        return "0"
    return r


def iprod_mat(elts):
    """elts: an iterable of elements with attribute iprod
    returns: the matrix with (ei.iprod(ej)) as its (i,j)-entry"""
    result = sp.zeros(len(elts))
    for i in range(len(elts)):
        c1 = elts[i]
        for j in range(i, len(elts)):
            c2 = elts[j]
            val = c1.iprod(c2)
            result[i, j] = val
            result[j, i] = val
    return result


def augment_with_identity(M):
    """M: a matrix
    returns: matrix (M|Id(n)), where n=shape(M)[0], the number of columns"""
    result = M
    for i in range(sp.shape(M)[0]):
        new_col = [0] * sp.shape(M)[0]
        new_col[i] = 1
        new_col = sp.Matrix([[k] for k in new_col])
        result = result.col_insert(sp.shape(M)[1] + i, new_col)
    return result


def Mat_preim_elt(M, v):
    """Returns a vector w so that M*w=v
    INPUTS:
    * 'M' - A matrix representing a finite dimensional linear operator
    * 'v' - a vector in the codomain of M
    """
    # This could also be done using pseudo-inverses (probably caching),
    # probably more quickly
    

    # We don't need to solve this entirely, just find a single preim_elt
    A = M.col_insert(sp.shape(M)[1], sp.Matrix(v))
    a = sp.symbols("a0:{num_symb}".format(num_symb=sp.shape(M)[1]))
    S = sp.solve_linear_system(A, *a)
    if S is None:
        return None
    b = [ai for ai in a if ai not in S]
    S1 = {}
    if len(b) != 0:
        S1[b[0]] = 1
        for i in range(1, len(b)):
            S1[b[i]] = 0
    return sp.Matrix([a]).transpose().subs(S).subs(S1)


def new_Mat_preim_elt(M, v):
    """Returns a vector w so that M*w=v
    INPUTS:
    * 'M' - A matrix representing a finite dimensional linear operator
    * 'v' - a vector in the codomain of M
    """
    # We don't need to solve this entirely, just find a single preim_elt
    A, cols = M.col_insert(sp.shape(M)[1], sp.Matrix(v)).rref()
    r = sp.zeros(sp.shape(M)[1], 1)
    for j in cols:
        i = 0
        while A[i, j] == 0:
            i += 1
        r[j] = A[i, -1]
    return r


def ZV(v):
    """returns true if V is a zero vector, False otherwise"""
    try:
        for i in v:
            if i != 0:
                return False
        return True
    except:
        return False


def col_sp_and_preim(M:sp.MatrixBase)->tuple[list[sp.MatrixBase],list[sp.MatrixBase]]:
    """arg: a Matrix M representing a finite dimensional linear operator
    returns: a pair of lists of coords (A,B) where
             A = elements of the domain which map to the elts of B
             B = basis for the columnspace of M"""
    result:tuple[list[sp.MatrixBase],list[sp.MatrixBase]] = ([], [])

    # transpose and augment
    T = sp.transpose(M)
    T = augment_with_identity(T)
    ## row reduce
    red = T.rref()
    T = red[0]

    piv = red[1]
    i = 0
    while piv[i] < sp.shape(M)[0]:
        result[0].append(T.row(i)[sp.shape(M)[0] : len(T.row(i))])
        result[1].append(T.row(i)[0 : sp.shape(M)[0]])
        i += 1
        if i == len(piv):
            break
    return result


def wedge_Mat(A:sp.MatrixBase, k:int)->sp.MatrixBase:
    """arg: A, a k-upper triangular matrix (k UT ==> k-1 UT, and UT <==> 0 UT)
    result: A\\wedge A as a matrix with lex r_ord on the basis
    (i.e., (1,2)<(1,3)<(1,4)<(2,3)<(2,4)<(3,4))"""
    n, m = sp.shape(A)  # n=#rows, m=#cols

    # Check that A is k-UT
    if not kUT_test(A, k):
        raise wedge_Mat_Exception(A, "Given matrix must be k-UT: ")

    # For fast index conversion, keep track of indices in a dict
    r_ord = {}
    ctr = 0
    for i in range(n):
        for j in range(i + 1, n):
            r_ord[(i, j)] = ctr
            ctr += 1

    c_ord = {}
    ctr = 0
    for i in range(m):
        for j in range(i + 1, m):
            c_ord[(i, j)] = ctr
            ctr += 1

    result = sp.SparseMatrix(sp.zeros(len(list(r_ord.keys())), len(list(c_ord.keys()))))

    # i1+k<=j1<i2+k<=j2
    for i1 in range(n):
        for j1 in range(max(i1 + k, 0), m):
            for i2 in range(j1 - k + 1, n):
                for j2 in range(i2 + k, m):
                    result[r_ord[(i1, i2)], c_ord[(j1, j2)]] = A[i1, j1] * A[i2, j2]

    # i1+k<i2+k<=j1<j2
    for i1 in range(n):
        for i2 in range(i1 + 1, n):
            for j1 in range(max(i2 + k, 0), m):
                for j2 in range(j1 + 1, m):
                    result[r_ord[(i1, i2)], c_ord[(j1, j2)]] = (
                        A[i1, j1] * A[i2, j2] - A[i2, j1] * A[i1, j2]
                    )
    return result


def kUT_test(A:sp.MatrixBase, k:int)->bool:
    """Tests if a matrix A is k-upper triangular"""
    for i in range(sp.shape(A)[0]):
        for j in range(0, min(i + k, sp.shape(A)[1])):
            if A[i, j] != 0:
                return False
    return True


def homog_exp(M:sp.MatrixBase, w:int, m:int)->sp.MatrixBase:
    """args: M, a square matrix of size 2m+5
          w, the graded weight of M
          m, such that len(T.basis) = 2m+5
    returns: exp(M)"""
    if w == 0:
        r = sp.diag(*[sp.exp(M[i, i]) for i in range(sp.shape(M)[0])])
        r[1:3, 1:3] = sp.exp(M[1:3, 1:3])
    if w > 0:
        s = (2 * m + 2) // w
        L = [sp.eye(2 * m + 5), M]
        for i in range(1, s + 1):
            L.append(wghted_mul(M, w, L[i], i * w, m))
        r = sum(
            [L[k] / sp.factorial(k) for k in range(len(L))], start=sp.zeros(2 * m + 5)
        )
    return r


def rand_UT(n:int, Min:int=0, Max:int=99, nonzero_entries:int|None=None)->sp.MatrixBase:
    R = sp.eye(n)
    if nonzero_entries is None:
        R = sp.randMatrix(n, min=Min, max=Max)
        for i in range(n):
            R[i, i] = 1
            for j in range(i):
                R[i, j] = 0
        return R

    R = sp.eye(n)
    ctr = 0
    while ctr <= nonzero_entries:
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)
        R[i, j] = random.randint(Min, Max)
        ctr += 1
    return R


def coords_to_lin_comb(basis:list[Union[sp.Symbol,sp.Indexed]], v:'VectorInput')->sp.Expr:
    """args: basis, a list of symbols, and coords, a vector of the same length
    Returns: A LinComb corresponding to the vector coords
    NOTE: basis must be a list of symbols"""

    # Check if the lengths are the same:
    coords=to_sympy_matrix(v)
    if len(basis) != len(coords):
        print("coords_to_lin_comb error: |basis| and |coords| have different lengths")
        return None

    result = 0
    for i in range(len(basis)):
        result = result + coords[i] * basis[i]
    return result
