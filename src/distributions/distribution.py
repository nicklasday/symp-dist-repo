__all__ = ["Distr_of_constant_symbol", "Compute_JS_dict"]
import shelve
import sympy as sp
import time

from ..utils.types import VectorInput
from ..utils.sympy_utils import to_sympy_matrix
from ..utils import math_helpers as mh
from ..utils import ds_helpers as dsh
import copy
from numbers import Number
from .diff_op_dict import DiffOpDict
from ..algebra.tanaka_symbols import TSymbElt,TSymb
from ..algebra.tanaka_symbols.tanaka_symbol_basis_elt import TSymbBasisElt
from ..algebra.tensor_algebras import TensorAlgElt
from ..algebra.complexes.cochain import Cochain
from .vector_field import Vector_Field


class Distr_of_constant_symbol(object):
    def __init__(self, symb:TSymb, curv:Cochain):
        """INPUTS:
        * 'symb' - a T_symb object
        * 'curv' - a positive 2-cochain from the cochain complex associated with symb"""
        self.Tanaka_symbol:TSymb = symb
        self.curv:Cochain = curv
        self.Jacobi_id_cache:dict[tuple[int,int,int,int],sp.Expr] = {}
        self.basis = [Vector_Field(A.vec, self) for A in symb.basis if A.wght < 0]
        self.normalized_diff_ops:list[dict[tuple[int,...],sp.Expr]] = [{}] * len(
            self.Tanaka_symbol.basis
        )  # coeff dictionary with keys tuple of ints and values coefficients

    def bracket(self, v1:Vector_Field, v2:Vector_Field)->Vector_Field:
        """Returns [v1,v2] as a section of self
        INPUTS:
        * 'v1','v2' - Vector_Field objects with self as parent
        """
        g1, g2 = [
            self.Tanaka_symbol.elt(sp.Matrix(v1.vec)),
            self.Tanaka_symbol.elt(sp.Matrix(v2.vec)),
        ]
        r = (g1.ad(g2)).vec
        e1e2 = g1.cast_as_ext_elt().wedge(g2.cast_as_ext_elt())
        r += self.curv.apply_cochain_map(e1e2).vec
        for i in range(len(v1.vec)):
            for j in range(len(v2.vec)):
                r[i] += v1.vec[j] * self.abn_ind_der(v2.vec[i], j) - v2.vec[
                    j
                ] * self.abn_ind_der(v1.vec[i], j)
        return Vector_Field(r, self)

    def Jacobi_id(self, i:int, j:int, k:int, l:int)->sp.Expr:
        # # Maybe I can speed this up by writing it out in terms of curvatures?
        # # It's got to be fast than what I'm doing here.

        # To do: speed this up!
        """Returns the Jacobi indentity in curvatures from cyc_{i,j,k}([[X_i,X_j],X_k]^l)"""
        if i == j or j == k or k == i:
            return 0

        if (i, j, k, l) in self.Jacobi_id_cache:
            return self.Jacobi_id_cache[(i, j, k, l)]
        if (k, i, j, l) in self.Jacobi_id_cache:
            return self.Jacobi_id_cache[(k, i, j, l)]
        if (j, k, i, l) in self.Jacobi_id_cache:
            return self.Jacobi_id_cache[(j, k, i, l)]

        ind_list = [i, j, k, i, j, k]
        r = 0

        B = self.Tanaka_symbol.basis

        def dwi(p, q, s):
            res1 = None
            res0 = None
            if q < p:
                res1 = -1
                res0 = self.curv.parent.dwi((str(B[q]), str(B[p]), str(B[s])))
            else:
                res1 = 1
                res0 = self.curv.parent.dwi((str(B[p]), str(B[q]), str(B[s])))
            return [res0, res1]

        def curv_coeff(p, q, s):
            if p == q:
                return 0
            result = 0
            t, sgn = dwi(p, q, s)
            try:
                result = result + sgn * self.curv.vd[t[0]][t[1]][t[2]]
            except:
                pass
            if B[p].wght + B[q].wght - B[s].wght == 0:
                result = result + B[p].ad(B[q]).vec[s]
            return result
            # # Missing the weight 0 terms

        for n in range(3):  # cyclic sum
            a, b, c = ind_list[n : n + 3]
            for d in range(3, len(self.basis) + 3):
                r = r + curv_coeff(a, b, d) * curv_coeff(d, c, l)
            r = r - self.normal_ind_der(curv_coeff(a, b, l), c)
        for n in range(3):  # cyclic sum
            a, b, c = ind_list[n : n + 3]
            self.Jacobi_id_cache[(a,b,c,l)]=r
        return r
        # To do: Check which of these is faster

        vi = Vector_Field(
            [0] * i + [1] + [0] * (len(self.Tanaka_symbol.basis) - i - 1), self
        )
        vj = Vector_Field(
            [0] * j + [1] + [0] * (len(self.Tanaka_symbol.basis) - j - 1), self
        )
        vk = Vector_Field(
            [0] * k + [1] + [0] * (len(self.Tanaka_symbol.basis) - k - 1), self
        )

        r = Vector_Field([0] * len(self.Tanaka_symbol.basis), self)
        ind_list = [vi, vj, vk, vi, vj, vk]

        for n in range(3):  # cyclic sum
            a, b, c = ind_list[n : n + 3]
            r += self.bracket(self.bracket(a, b), c)
        i_list = [i, j, k]
        i_list.sort()
        for l1 in range(len(self.Tanaka_symbol.basis)):
            self.Jacobi_id_cache[tuple(i_list + [l1])] = self.normalize_der(r.vec[l1])
        return self.normalize_der(r.vec[l])

    @classmethod
    def abn_ind_der(self, ind_expr:sp.Expr, i:int)->sp.Expr:
        """Returns the derivative of ind_expr in the direction X_i among X_3, X_4,..., X_{2n-1},
        which is a frame on the base manifold.

        Note: This only works for horizontal derivatives and for symplectic symbols!

        INPUTS:
        * 'ind_expr' -- an expression in h, e, y, and indexed objects
        * 'i' -- an integer between 3 and 2n-1
        """
        if isinstance(ind_expr,sp.MatrixBase):
            if type(ind_expr) == sp.ImmutableDenseMatrix:
                result = mh.mut_mat_copy(ind_expr)
            else:
                result = copy.copy(ind_expr)
            for j in range(sp.shape(result)[0]):
                for k in range(sp.shape(result)[1]):
                    result[j, k] = self.abn_ind_der(result[j, k], i)
            return result

        if isinstance(ind_expr, TSymbBasisElt):
            return ind_expr.parent.elt()
        if isinstance(ind_expr,TSymbElt):
            v = [self.abn_ind_der(ind_expr.vec[j], i) for j in range(len(ind_expr.vec))]
            return ind_expr.parent.elt(v)
        if isinstance(ind_expr, Number):
            return 0
        if isinstance(ind_expr,sp.Add):
            result = sp.Add(*[self.abn_ind_der(A, i) for A in ind_expr.args])
            return result
        if isinstance(ind_expr,sp.Mul):
            result = 0
            for j in range(len(ind_expr.args)):
                result += self.abn_ind_der(ind_expr.args[j], i) * sp.Mul(
                    *list(
                        ind_expr.args[0:j] + ind_expr.args[j + 1 : len(ind_expr.args)]
                    )
                )
            return result
        if isinstance(ind_expr,sp.Pow):
            return (
                ind_expr.exp
                * ind_expr.base ** (ind_expr.exp - 1)
                * self.abn_ind_der(ind_expr.base, i)
            )
        if isinstance(ind_expr,sp.Indexed):
            base = ind_expr.base
            ind = list(ind_expr.indices)
            return base[ind + [i]]
        if isinstance(ind_expr,sp.Symbol):
            return 0
        if isinstance(ind_expr,sp.exp):
            return ind_expr * self.abn_ind_der(ind_expr.args[0], i)

    def normalized_diff_op(self, i:int)->dict[tuple[int,...],sp.Expr]:
        if self.normalized_diff_ops[i] is {}:
            self.set_normalized_diff_op(i)
        return self.normalized_diff_ops[i]

    def set_normalized_diff_op(self, i:int)->None:
        """sets the self.normalized_diff_ops[i], which is a dictionary d representing the X_i derivative.
        The keys of d are integer tuples representing iterated fundamental derivatives, and the values
        of d are coefficients."""
        r:dict[tuple[int,...],sp.Expr] = {}
        b = sp.IndexedBase("beta")

        # Normalize beta[0,0,0,i]
        d = sp.expand(
            self.normalize_der(self.abn_ind_der(b[0, 0, 0], i))
        ).as_coefficients_dict()
        for k in d:
            # Distinguish the coeff (which don't involve b) from the b factors
            coeff = k.as_coeff_mul()[0] * d[k]
            key:tuple[int,...]
            one_b_term=False
            for A in k.as_coeff_mul()[1]:
                # Exactly one term of the product A will have base b
                # all others will be (possibly) curvature terms
                if isinstance(A,sp.Indexed) and A.base == b:
                    key = A.indices[3 : len(A.indices)]
                    one_b_term=not one_b_term
                else:
                    coeff = coeff * A
                if not one_b_term:
                    raise AssertionError("Unreachable code reached")
            if key in r:
                r[key] = r[key] + coeff
            else:
                r[key] = coeff
        self.normalized_diff_ops[i] = r

    def normal_ind_der(self, expr:sp.Expr, i:int)->sp.Expr:
        """Returns the derivative of ind_expr in the direction X_i among X_3, X_4,..., X_{2n-1},
        which is a frame on the base manifold, expressed as commutators of those VFs from the -1 piece.

        Note: This only works for horizontal derivatives!

        INPUTS:
        * 'ind_expr' -- an expression in h, e, y, and indexed objects
        * 'i' -- an integer between 3 and 2n-1"""

        r = 0
        NDO = self.normalized_diff_op(i)
        for k in NDO:
            coeff = NDO[k]
            temp = copy.copy(expr)
            for i in k:
                temp = self.abn_ind_der(temp, i)
            r += coeff * temp
        return r

    def are_deriv_normal(self, expr:sp.Expr)->bool:
        """Returns True if all the directional derivatives of Indexed objects which appear in expr
        are in normal form; that is, they are expressed using only the fundamental directions.
        INPUTS:
        * 'expr' - an algebraic expression in Indexed objects and symbols"""
        if type(expr) == TSymbElt:
            for a in expr.vec:
                if not self.are_deriv_normal(a):
                    return False
            return True
        if issubclass(type(expr), TensorAlgElt):
            for k1 in expr.vd:
                for k2 in expr.vd[k1]:
                    if not self.are_deriv_normal(expr.vd[k1][k2]):
                        return False
            return True
        if type(expr) == sp.Matrix:
            for i in range(sp.shape(expr)[0]):
                for j in range(sp.shape(expr)[1]):
                    if not self.are_deriv_normal(expr[i, j]):
                        return False
            return True
        if type(expr) in [list, tuple]:
            for a in expr:
                if not self.are_deriv_normal(a):
                    return False
            return True

        s = mh.Indexed_obj_in_expr(expr)
        for A in s:
            if len(A.indices) > 3:
                for i in A.indices[3 : len(A.indices)]:
                    if self.Tanaka_symbol.basis[i].wght < -1:
                        return False
        return True

    def normalize_der(self, expr:sp.Expr)->sp.Expr:
        """Converts all derivatives to normal form using the bracket relations from self. Normal form means all derivatives
        are in the fundamental directions, directions 3 and 4.

        WARNING: This assumes all IndexedBase objects have 3 indices which are not derivatives (like curvature coefficients)
        INPUTS:
        * 'expr' - an algebraic expression in Indexed objects and symbols

        WARNING: This only works for distributions with the symplectic symbol"""

        if self.are_deriv_normal(expr):
            return expr

        if type(expr) == TSymbElt:
            return expr.parent.elt([self.normalize_der(a) for a in expr.vec])
        if issubclass(type(expr), TensorAlgElt):
            new_vd:dict[int,dict[int,sp.MatrixBase]] = {}
            for k1 in expr.vd:
                new_vd[k1] = {}
                for k2 in expr.vd[k1]:
                    new_vd[k1][k2] = self.normalize_der(expr.vd[k1][k2])
            return expr.parent.elt(new_vd)
        if type(expr) in [
            sp.Matrix,
            sp.SparseMatrix,
            sp.ImmutableDenseMatrix,
            sp.MutableDenseMatrix,
        ]:
            r = sp.zeros(*sp.shape(expr))
            for i in range(sp.shape(expr)[0]):
                for j in range(sp.shape(expr)[1]):
                    r[i, j] = self.normalize_der(expr[i, j])
            return r
        if isinstance(expr, Number):
            return expr
        if isinstance(expr,sp.Add):
            return sp.Add(*[self.normalize_der(A) for A in expr.args])
        if isinstance(expr,sp.Mul):
            return sp.Mul(*[self.normalize_der(A) for A in expr.args])
        if isinstance(expr,sp.Pow):
            return self.normalize_der(expr.base) ** self.normalize_der(expr.exp)
        if isinstance(expr,sp.Symbol):
            return expr
        if isinstance(expr,sp.exp):
            return sp.exp(self.normalize_der(expr.args[0]))
        if isinstance(expr,sp.Indexed):
            # Find the first abnormal direction k
            b = expr.base
            prefix = list(expr.indices[0:3])
            i = 3
            while i < len(expr.indices) and expr.indices[i] in [3, 4]:
                prefix.append(expr.indices[i])
                i += 1
            k = expr.indices[len(prefix)]
            postfix = list(expr.indices[len(prefix) + 1 : len(expr.indices)])

            # [Xi,Xj] = Xk in the Tanaka symbol
            if k in range(5, len(self.Tanaka_symbol.basis) - 1):
                i, j = (3, k - 1)
            if k == len(self.Tanaka_symbol.basis) - 1:
                i, j = (5, len(self.Tanaka_symbol.basis) - 3)

            no_post_r = (
                b[prefix + [j, i]] - b[prefix + [i, j]]
            )  # Highest order terms, no postfix

            # Kurvs=(Xk-[Xi,Xj]).vec
            Xi = Vector_Field(
                [0] * (i) + [1] + [0] * (len(self.Tanaka_symbol.basis) - i - 1), self
            )
            Xj = Vector_Field(
                [0] * (j) + [1] + [0] * (len(self.Tanaka_symbol.basis) - j - 1), self
            )
            Xk = Vector_Field(
                [0] * (k) + [1] + [0] * (len(self.Tanaka_symbol.basis) - k - 1), self
            )
            Neg_Kurvs = (Xk - self.bracket(Xi, Xj)).vec
            for q in range(len(Neg_Kurvs)):
                no_post_r += Neg_Kurvs[q] * b[prefix + [q]]

            r = self.normalize_der(no_post_r)
            for a in postfix:
                r = self.normalize_der(self.abn_ind_der(r, a))
            # Before substituting with Jacobi, I should make sure the values of Jacobi
            # have only normal derivatives
            return r
        print("normalize_der received unhandled type", type(expr))

    def dir_der(self, f:sp.Expr, v:VectorInput)->sp.Expr:
        """Returns the directional derivative of f in the direction v"""
        u = v
        if isinstance(v,sp.MatrixBase):
            u = list(v)
        if isinstance(v, Vector_Field):
            u = list(v.vec)

        if isinstance(f, list):
            return [self.dir_der(a, v) for a in f]

        if isinstance(f, sp.Matrix):
            r = sp.zeros(*sp.shape(f))
            for i in range(sp.shape(f)[0]):
                for j in range(sp.shape(f)[1]):
                    r[i, j] = self.dir_der(f[i, j], v)
            return r

        r = 0
        g = self.Tanaka_symbol
        for i in range(len(g.basis) - len(g.m_basis), len(u)):
            if u[i] != 0:
                r += u[i] * self.abn_ind_der(f, i)
        return r

    def der_term(self, i0:int, i1:int, i2:int, i3:int)->sp.Expr:
        """Returns the derivative term for the component of the Bianchi identity
        applied to (i0,i1,i2), which is an element of self.Tanaka_symbol."""
        g = self.Tanaka_symbol
        X0, X1, X2 = [g.basis[a] for a in [i0, i1, i2]]
        w0, w1, w2 = [None, None, None]
        try:
            w0 = g.ext_alg.elt_from_cd({(str(X1), str(X2)): 1})
        except KeyError:
            pass
        try:
            w1 = g.ext_alg.elt_from_cd({(str(X2), str(X0)): 1})
        except KeyError:
            pass
        try:
            w2 = g.ext_alg.elt_from_cd({(str(X0), str(X1)): 1})
        except KeyError:
            pass

        r = 0
        if w0 is not None:
            r += self.normal_ind_der(self.curv.apply_cochain_map(w0).vec[i3], i0)
        if w1 is not None:
            r += self.normal_ind_der(self.curv.apply_cochain_map(w1).vec[i3], i1)
        if w2 is not None:
            r += self.normal_ind_der(self.curv.apply_cochain_map(w2).vec[i3], i2)
        return -r  # negative because of differing sign conventions

    def cb_term(self, i0:int, i1:int, i2:int, i3:int)->sp.Expr:
        """returns the coboundary of K applied to i0,i1,i2 columns of F.
        This is needed because we care about the value as a cochain in C(g,g), not just C(m,g)
        (at least for the purpose of checks)"""
        r = self.Tanaka_symbol.elt()
        X0, X1, X2 = [self.Tanaka_symbol.basis[a] for a in [i0, i1, i2]]

        for i in range(3):
            Y0, Y1, Y2 = [X0, X1, X2, X0, X1][i : i + 3]
            w0 = (
                Y0.negative_projection()
                .cast_as_ext_elt()
                .wedge(Y1.negative_projection().cast_as_ext_elt())
            )
            r += -self.curv.apply_cochain_map(w0).ad(Y2)
            w1 = (
                Y0.ad(Y1)
                .negative_projection()
                .cast_as_ext_elt()
                .wedge(Y2.negative_projection().cast_as_ext_elt())
            )
            r += -self.curv.apply_cochain_map(w1)
        return -r.vec[i3]  # negative because of differing sign conventions

    def gerst_term(self, i0:int, i1:int, i2:int, i3:int)->sp.Expr:
        X0, X1, X2 = [self.Tanaka_symbol.basis[a] for a in [i0, i1, i2]]
        r = self.Tanaka_symbol.elt()
        for i in range(3):
            Y0, Y1, Y2 = [X0, X1, X2, X0, X1][i : i + 3]
            Y0 = Y0.negative_projection().cast_as_ext_elt()
            Y1 = Y1.negative_projection().cast_as_ext_elt()
            Y2 = Y2.negative_projection().cast_as_ext_elt()
            w = (
                self.curv.apply_cochain_map(Y0.wedge(Y1))
                .negative_projection()
                .cast_as_ext_elt()
            )
            r += self.curv.apply_cochain_map(w.wedge(Y2))
        return r.vec[i3]

    def Bianchi(self, i0:int, i1:int, i2:int, i3:int, subdivide=False)->sp.Expr|tuple[sp.Expr,sp.Expr,sp.Expr]:
        """Returns the Bianchi expression corresponding to indices (i,j,k,l) in the frame F.
        Should be zero modulo the Jacobi identity for the downstairs distribution"""
        r0 = self.cb_term(i0, i1, i2, i3)
        r1 = self.der_term(i0, i1, i2, i3)
        r2 = self.gerst_term(i0, i1, i2, i3)
        if subdivide:
            return (r0, r1, r2)
        return r0 + r1 + r2

    def Ricci_Id(self, i:int, j:int)->DiffOpDict:
        """returns the Ricci identity for self corresponding to indices i and j as a DiffOpDict"""
        if i == j:
            return DiffOpDict({}, self.Tanaka_symbol, self.curv)
        r = DiffOpDict({(i, j): 1, (j, i): -1}, self.Tanaka_symbol, self.curv)
        vi = Vector_Field(sp.eye(len(self.Tanaka_symbol.basis)).col(i), self)
        vj = Vector_Field(sp.eye(len(self.Tanaka_symbol.basis)).col(j), self)
        w = self.bracket(vi, vj)
        for k in range(len(w.vec)):
            r.d[(k,)] = -w.vec[k]
        r.clear_zeros()
        return r.normal_form()


def Compute_JS_dict(
    Distr:Distr_of_constant_symbol, max_wght:int, unshelve:bool=False, reshelve:bool=False, name:str|None=None, constant_coeff:bool=False
)->tuple[dict[sp.Indexed,dict[tuple[int,...],sp.Expr]],list[sp.Expr]]:
    """Returns a differntial substitution dictionary representing the
    Jacobi substitutions for the given distribution up to and including max_wght

    INPUTS:
    * 'Distr' - A distribution of constant symbol
    * 'max_wght' - a natural number weight
    * 'unshelve' - if True, will use shelved values from file 'Jacobi_ideals'
    * 'reshelve' - if True, will shelve result to 'Jacobi_ideals'
    * 'name' - a title for the distribution; required if unshelving or reshelving"""
    if (unshelve or reshelve) and name is None:
        raise ValueError(
            "Compute_JS_dict: argument name is required when unshelving or reshelving"
        )

    g = Distr.Tanaka_symbol
    result:dict[sp.Indexed,dict[tuple[int,...],sp.Expr]] = {}
    p = g.basis.index(g.m_basis[0])  # index of first negative element

    if constant_coeff:
        for a in mh.Indexed_obj_in_expr(Distr.curv):
            result[a] = {}
            for i in range(len(g.basis)):
                if g.basis[i].wght == -1:
                    result[a][(i,)] = 0

    # First, sort the Jacobi indices by weight
    JI_ind_wghts:dict[int,list[tuple[int,int,int,int]]] = {}
    for i in range(p, len(g.basis)):
        for j in range(i + 1, len(g.basis)):
            for k in range(j + 1, len(g.basis)):
                for l in range(p, len(g.basis)):
                    w = (
                        -g.basis[i].wght
                        - g.basis[j].wght
                        - g.basis[k].wght
                        + g.basis[l].wght
                    )
                    if w not in JI_ind_wghts:
                        JI_ind_wghts[w] = []
                    JI_ind_wghts[w].append((i, j, k, l))

    # If unshelving is available, return that.
    if unshelve and name is not None:
        with shelve.open("Jacobi_ideals") as shelf:
            try:
                return shelf[name + "{j}".format(j=max_wght)]
            except:
                pass

    # if this weight hasn't been shelved, compute it,
    # possibly unshelving along the way
    not_added:list[sp.Expr] = []
    for i in range(1, max_wght + 1):
        time0 = time.time()
        for j in range(len(JI_ind_wghts[i])):
            ind = JI_ind_wghts[i][j]
            dsh.process_JI(ind, Distr, result, not_added=not_added)
        if reshelve and name is not None:
            with shelve.open("Jacobi_ideals") as shelf:
                shelf[name + "{j}".format(j=i)] = (result, not_added)
        print(
            "Jacobi id of weight",
            i,
            "computed in time",
            mh.hrs_min_sec(time.time() - time0),
        )
    return (result, not_added)
