__all__ = ["Geom_Prolongation"]

import copy
import sympy as sp
import time
from ..utils import math_helpers as mh
from ..utils import lin_alg_helpers as lh
from ..algebra.tanaka_symbols import TSymbElt
from ..utils import ds_helpers as dsh
from ..distributions.distribution import Distr_of_constant_symbol
from ..algebra.tanaka_symbols import SympSymb


class Geom_Prolongation(object):
    def __init__(self, D:Distr_of_constant_symbol, J_subs_dict:dict[sp.Indexed,dict[tuple[int,...],sp.Expr]]):
        """Constructs a geometric prolongation of the distribution D
        which has Tanaka symbol given by the symplectic algebra"""
        self.alg:SympSymb = D.Tanaka_symbol
        self.distr = D
        self.jsd = J_subs_dict
        self.cochain_complex = self.alg.cochain_complex

        # # The Adjoint matrix corresponding to a point p=(y,h,e)
        self.Ad_p = sp.exp(
            sp.symbols("h") * self.alg.basis[1].ad_mat()
            + sp.symbols("e") * self.alg.basis[2].ad_mat()
        ) * sp.exp(sp.symbols("y") * self.alg.basis[0].ad_mat())
        self.Ad_p_inv = self.Ad_p.inv()

        self.K = sp.IndexedBase("K")
        self.gf = [None]  # Temporary for testing
        self.gdf = [None]  # Temporary for testing
        self.pK = [None]
        self.KN = self.cochain_complex.elt({})
        self.FN = sp.eye(len(self.alg.basis))
        self.FN_inv = sp.eye(len(self.alg.basis))
        self.fk = self.cochain_complex.elt({})
        self.harm_curv_cache = [
            0,
            self.cochain_complex.elt({}),
        ]  # (max_weight, harm_curv)
        self.norm_wght = 0

        self.m = sp.Integer((len(self.alg.basis) - 5) / 2)
        self.yhe_pt = self.alg.elt(
            [sp.symbols("y"), sp.symbols("h"), sp.symbols("e")] + [0] * (len(self.alg.basis) - 3)
        )
        # This line sometimes takes a bit of time
        self.Omega_1_curv = self.alg.second_kind_inv(self.yhe_pt).Ad(
            self.distr.curv, mod="CE"
        )

        # changes the frame from Omega_1 to dy, dh, de and horizontal VFs
        self.cof_mat = self.change_of_frame_mat()

    def change_of_frame_mat(self):
        # change the frame from Omega_1 to fundamental VFs and horizontal VFs
        cof1 = sp.eye(len(self.alg.basis))
        cof1[3 : len(self.alg.basis), 3 : len(self.alg.basis)] = self.Ad_p[
            3 : len(self.alg.basis), 3 : len(self.alg.basis)
        ]
        temp1 = sp.zeros(len(self.alg.basis))
        temp1[0:3, 0 : len(self.alg.basis)] = self.Ad_p[0:3, 0 : len(self.alg.basis)]
        cof1[0:3, 3 : len(self.alg.basis)] = (self.Ad_p_inv * temp1)[
            0:3, 3 : len(self.alg.basis)
        ]

        # change of frame matrix from fundamental and horizontal to dy, dh, de and horizontal
        cof2 = sp.eye(len(self.alg.basis))
        cof2[0, 1] = 2 * sp.symbols("y")
        return cof2 * cof1

    def dir_der(self, f, v, normal=False):
        """Returns the directional derivative of f in the direction of v
        INPUTS:
        * 'f' - an expression in h,e,y, and K
        * 'v' - a vector representation in Omega_1 of a vector field on P^1
        """
        if type(f) == TSymbElt:
            return f.parent.elt([self.dir_der(a, v) for a in f.vec])
        if type(f) in [sp.Matrix, sp.SparseMatrix]:
            r = sp.zeros(*sp.shape(f))
            for i in range(sp.shape(f)[0]):
                for j in range(sp.shape(f)[1]):
                    r[i, j] = self.dir_der(f[i, j], v)
            return r

        y, h, e = sp.symbols("y,h,e")
        dy = sp.diff(f, y)
        dh = sp.diff(f, h)  # It's slightly inefficient to compute these every time, but not a huge deal (probably)
        de = sp.diff(f, e)

        der_vec = [dy, dh, de] + [0] * (len(self.alg.basis) - 3)

        # self.cof_mat is change of frame from Omega_1 to vertical and horizontal derivatives
        cof_v = self.cof_mat * v
        for i in range(3, len(v)):
            if cof_v[i] != 0:
                if normal:
                    der_vec[i] = self.distr.normal_ind_der(f, i)
                else:
                    der_vec[i] = self.distr.abn_ind_der(f, i)
        der_vec = sp.Matrix([der_vec])

        return (der_vec * cof_v)[0, 0]

    def fund_der(self, f, *dir_list, F):
        r"""Returns the fundamental derivative of f in the directions of dir_list.

        args:
        * 'f' - an expression in h, e, y, and K
        * 'dir_list' - a list of indices which refer to the colums of F
        * 'F' - the matrix expression of a frame in \tilde\Omega_1
        """
        r = copy.copy(f)
        for i in dir_list:
            r = self.dir_der(f, F.col(i))
        return r

    def Omega_1_bracket(self, Y1, Y2, min_w=-sp.oo):
        """Freezes coefficients, then returns the components of [Y1,Y2] in weight >=min_w.

        INPUTS:
        * 'Y1','Y2' -- vector reps in Omega_1 of vector fields on P1;
          entries may involve h,e,y and K (which are considered constant)
        """
        r = self.alg.elt(Y1).ad(self.alg.elt(Y2)).vec
        # Curvature must be added to r
        Z1 = self.alg.elt([0, 0, 0] + Y1[3 : len(Y1)])
        Z2 = self.alg.elt([0, 0, 0] + Y2[3 : len(Y2)])
        ext_elt = Z1.cast_as_ext_elt().wedge(Z2.cast_as_ext_elt())
        r = r + self.Omega_1_curv.apply_cochain_map(ext_elt).vec
        for i in range(len(r)):
            if self.alg.basis[i].wght < min_w:
                r[i] = 0
        return r

    def bracket(self, Y1, Y2, min_w=-sp.oo):
        """Returns the components of [Y1,Y2] of weight at least min_w for the vector fields Y1, Y2 on P^1

        INPUTS:
        * 'Y1','Y2' -- vector reps in Omega_1 of vector fields on P1; entries may involve h,e,y and K
        * 'min_w' -- an integer weight
        """
        r = sp.zeros(len(self.alg.basis), 1)
        for i in range(len(self.alg.basis)):
            if self.alg.basis[i].wght >= min_w:
                r[i] = [self.dir_der(Y2[i], Y1) - self.dir_der(Y1[i], Y2)]
        return r + self.Omega_1_bracket(Y1, Y2, min_w)

    def curv(self, F, l_curv, k):
        """Computes the curvature function for frame F on P^1 in weight precisely k.
        The curvature function is an element of C^2_+(m,g).
        INPUTS:
        * 'F', a block unipotent lower triangular matrix, representing in Omega_1 of a frame on P^1
        * 'l_curv' -- the curvature of F in wght < k as a cochain
        * 'k' -- natural number
        """
        B = self.alg.basis
        if k <= 0:
            return self.cochain_complex.elt({})

        r = {}
        for i in range(3, len(B)):
            for j in range(i + 1, len(B)):
                Z = self.bracket(F.col(i), F.col(j), B[i].wght + B[j].wght + k) - F * (
                    l_curv.apply_cochain_map_base(B[i], B[j]).vec + B[i].ad_mat().col(j)
                )
                # Z has entries in wght w(i)+w(j)+k
                s_ind = [
                    a for a in range(len(B)) if B[a].wght == B[i].wght + B[j].wght + k
                ]
                r[(i, j)] = sp.zeros(len(self.alg.basis), 1)
                for s in s_ind:
                    r[(i, j)][s] = Z[s]
        return self.cochain_complex.curv_dict_to_cochain(r)

    def normal_frame(self, k):
        """Returns a list [F,c], a frame on self which is normal up to degree k along with a cochain
        representing its curvature function up to degree k
        INPUTS:
        * 'k' -- a natural number
        """
        C = self.cochain_complex
        if k <= self.norm_wght:
            return (self.FN, self.KN, self.fk)
        elif k > self.norm_wght + 1:
            self.normal_frame(k - 1)

        # Now set weight k assuming (k-1) is already done
        time0 = time.time()
        # Compute the wght k curvature of the previous frame
        self.pK.append(self.KN + self.curv(self.FN, self.KN, k))
        time1 = time.time()
        print("weight", k, "curvature computed in time", mh.hrs_min_sec(time1 - time0))

        # project onto im(d) along the section,
        hey_z = {sp.symbols("y"): 0, sp.symbols("h"): 0, sp.symbols("e"): 0}
        dfk = -C.subspace_proj(self.pK[k].subs(hey_z), "exact")  # along the section
        time2 = time.time()
        print("weight", k, "projection computed in time", mh.hrs_min_sec(time2 - time1))

        # Choose fk from the preimage for normalization
        fk = C.cb_preim_elt(dfk,check=True)  # along the 0-section, for modifying the frame
        self.fk = fk
        time3 = time.time()
        print(
            "weight",
            k,
            "preimage cochain computed in time",
            mh.hrs_min_sec(time3 - time2),
        )
        # ##------------begin deprecated code------------

        # gdfk = self.alg.second_kind_inv(self.yhe_pt).Ad(dfk, mod="CE")
        # gfk = self.alg.second_kind_inv(self.yhe_pt).Ad(fk, mod="CE")  # I think this will give a Cartan connection...
        # self.KN = self.pK[k] + gdfk.wght_proj(k)  
        # # I think this should be "equivariant"...equivariant in the lowest nonzero weight
        # # since wght 3 KN is not equivariant, one of the above is not equivariant

        # # # temporary for testing
        # # self.gf.append(gfk)
        # # self.gdf.append(gdfk)

        # # modify the frame
        # step = int((self.alg.wght_list[0] - self.alg.wght_list[-1]) / k + 1)
        # gfk_equiv_mat = sp.simplify(C.one_cochain_mat_rep(gfk))
        # exp_gfk, exp_gfk_inv = lh.nilp_exp(gfk_equiv_mat, step)

        # ##------------end deprecated code------------

        ##------------begin new code------------
        step = int((self.alg.wght_list[0] - self.alg.wght_list[-1]) / k + 1)
        exp_fk,exp_fk_inv=lh.nilp_exp(sp.simplify(C.one_cochain_mat_rep(fk)), step)

        gdfk = self.alg.second_kind_inv(self.yhe_pt).Ad(dfk, mod="CE")
        self.KN = self.pK[k] + gdfk.wght_proj(k)  
        # I think this should be "equivariant"...equivariant in the lowest nonzero weight
        # since wght 3 KN is not equivariant, one of the above is not equivariant

        # modify the frame
        # to do: Fix this...Ad_mat isn't implemented for the cochain action
        yhe_Ad=self.yhe_pt.Ad_mat("CE",2,k)
        yhe_inv_Ad=self.alg.second_kind_inv(self.yhe_pt).Ad_mat("CE",2,k)
        exp_gfk=yhe_inv_Ad*exp_fk*yhe_Ad
        exp_gfk_inv = yhe_inv_Ad*exp_fk_inv*yhe_Ad
        ##------------end new code------------

        self.FN = self.FN * exp_gfk
        self.FN_inv = exp_gfk_inv * self.FN_inv

        time4 = time.time()
        print(
            "weight",
            k,
            "Matrix exponential complete in time",
            mh.hrs_min_sec(time4 - time3),
        )
        self.norm_wght = k

        self.KN = dsh.ds_subs(self.distr.normalize_der(self.KN), self.jsd)[
            0
        ]
        self.FN = dsh.ds_subs(self.FN, self.jsd)[0]
        time5 = time.time()
        print(
            "weight",
            k,
            "differential substitutions complete in time",
            mh.hrs_min_sec(time5 - time4),
        )

        print(
            "weight",
            k,
            "normalization complete in time",
            mh.hrs_min_sec(time.time() - time0),
        )
        return (self.FN, self.KN, fk)

    def der_term(self, F, K, i0, i1, i2, i3):
        """Returns the derivative term for the component of the Bianchi identity
        applied to (i0,i1,i2), which is an element of self.alg."""
        g = self.alg
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
            r += self.dir_der(K.apply_cochain_map(w0).vec[i3], F.col(i0))
        if w1 is not None:
            r += self.dir_der(K.apply_cochain_map(w1).vec[i3], F.col(i1))
        if w2 is not None:
            r += self.dir_der(K.apply_cochain_map(w2).vec[i3], F.col(i2))
        return -r

    def cb_term(self, F, K, i0, i1, i2, i3):
        """returns the coboundary of K applied to i0,i1,i2 columns of F.
        This is needed because we care about the value as a cochain in C(g,g), not just C(m,g)
        (at least for the purpose of checks)"""
        r = self.alg.elt()
        X0, X1, X2 = [self.alg.basis[a] for a in [i0, i1, i2]]

        for i in range(3):
            Y0, Y1, Y2 = [X0, X1, X2, X0, X1][i : i + 3]
            w0 = (
                Y0.negative_projection()
                .cast_as_ext_elt()
                .wedge(Y1.negative_projection().cast_as_ext_elt())
            )
            r += K.apply_cochain_map(w0).ad(Y2)
            w1 = (
                Y0.ad(Y1)
                .negative_projection()
                .cast_as_ext_elt()
                .wedge(Y2.negative_projection().cast_as_ext_elt())
            )
            r += K.apply_cochain_map(w1)
        return r.vec[i3]

    def gerst_term(self, F, K, i0, i1, i2, i3):
        X0, X1, X2 = [self.alg.elt(F.col(a)) for a in [i0, i1, i2]]
        r = self.alg.elt()
        for i in range(3):
            Y0, Y1, Y2 = [X0, X1, X2, X0, X1][i : i + 3]
            print("Y0 =", Y0)
            print("Y1 =", Y1)
            print("Y2 =", Y2)
            Y0 = Y0.negative_projection().cast_as_ext_elt()
            Y1 = Y1.negative_projection().cast_as_ext_elt()
            Y2 = Y2.negative_projection().cast_as_ext_elt()
            print(
                "K.apply_cochain_map(Y0.wedge(Y1)) =", K.apply_cochain_map(Y0.wedge(Y1))
            )
            w = (
                K.apply_cochain_map(Y0.wedge(Y1))
                .negative_projection()
                .cast_as_ext_elt()
            )
            print("w =", w)
            r += K.apply_cochain_map(w.wedge(Y2))
        return r.vec[i3]

    def Bianchi(self, F, K, i0, i1, i2, i3, subdivide=False):
        """Returns the Bianchi expression corresponding to indices (i,j,k,l) in the frame F.
        Should be zero modulo the Jacobi identity for the downstairs distribution"""
        r0 = self.cb_term(F, K, i0, i1, i2, i3)
        r1 = self.der_term(F, K, i0, i1, i2, i3)
        r2 = self.gerst_term(F, K, i0, i1, i2, i3)
        if subdivide:
            return (r0, r1, r2)
        return self.distr.normalize_der(r0 + r1 + r2)


# ## Prenormalization

# From the involutivity conditions $[V_i,V_i]\subseteq V_i$ and $[\mathcal{J}^{(i)},V_i]\subseteq \mathcal{J}^{(i)}$, we can impose the following conditions on the structure function $K_{ij}^k$:
#
# $$ [\varepsilon_i,\varepsilon_j]=\sum_{k=1}^j K_{ij}^k\varepsilon_k,\quad \text{for}\ i<j\leq m$$
# $$ [\varepsilon_i,\varepsilon_j]=K_{ij}^XX+\sum_{k=1}^j K_{ij}^k\varepsilon_k,\quad \text{for}\ j>m\ \text{and}\ i+j<2m+1$$
# All this is captured in the below "prenormalization", but the indices are shifted to reflect the index of each element in the basis $(Y,H,E,X,\varepsilon_1,\ldots \varepsilon_6,\eta)$.
