from __future__ import annotations
from typing import TYPE_CHECKING, Type
import sympy as sp
from itertools import combinations

from ...utils.exceptions import Coordinatization_Exception, Failed_Check_Exception


from ..tensor_algebras import TensorAlg
from .cochain import Cochain
from ...utils import math_helpers as mh
from ...utils import math_helpers as lh
from ...utils import lin_alg_helpers as lh
from ...utils import cochain_helpers as ch
from .cochain_basis_elt import CochainBasisElt, Apply_Gerst_prod_on_base

if TYPE_CHECKING:
    from ...algebra.tanaka_symbols.tanaka_symbol import TSymb
    from ...algebra.exterior_algebras.exterior_alg import ExtAlg
    from ..tensor_algebras.tensor_alg_elt import TensorAlgElt

__all__ = ["CochainComplex", "Gerst_prod", "Gerst_bracket"]


class CochainComplex(TensorAlg[Cochain]):
    def __init__(self, T_symb_obj:TSymb):
        T_symb_obj.cochain_complex = self
        super().__init__(T_symb_obj)
        self.alg:TSymb = T_symb_obj
        self.ext_alg:ExtAlg = T_symb_obj.ext_alg
        self.cb_mat_cache:dict[tuple[int,int],sp.MatrixBase] = {} # indexed by source
        self.cb_preim_mat_cache:dict[tuple[int,int],sp.MatrixBase] = {} 
        self.pinv_cache:dict[tuple[str,int,int],sp.MatrixBase]={}
        self.subspace_cache:dict[tuple[str,int,int],sp.MatrixBase]= {}
        self.rnc:Cochain|None = None # This represents an arbitrary regular normal cochain

    @property
    def childcls(self) -> Type[Cochain]:
        return Cochain 

    def tuple_wght(self, t:tuple[str,...])->int:
        """Returns the weight of t
        INPUTS:
        * 't' - a tuple representing an element of self
        """
        r = 0
        for i in range(len(t) - 1):
            j = self.alg.basis_strs.index(t[i])
            r = r - self.alg.wght_list[j]  # This is the exterior algebra of m_dual
        r += self.alg.wght_list[self.alg.basis_strs.index(t[-1])]
        return r

    def tuple_deg(self, t:tuple[str,...])->int:
        """Returns the degree of t
        INPUTS:
        * 't' - a tuple representing an element of self
        """
        return len(t) - 1

    def dwi(self, basis_str_tuple:tuple[str,...])->tuple[str,str,str]:
        """Returns the degree, weight, and index of basis_tuple
        Inputs:
        * 'basis_tuple' -- a tuple of basis strings representing a basic cochain
        """
        if basis_str_tuple not in self.dwi_dicts:
            self.init_basis(len(basis_str_tuple) - 1)
        return self.dwi_dicts[basis_str_tuple]

    def sort_tuple(self, t):
        """Returns a (cochain)sorting of t and the sign of the corresponding permutation, as a tuple
        INPUTS:
        * 't' - a tuple of basis_strs
        """
        temp = t[0:-1]
        if len(temp) != len(set(temp)):
            return (None, 0)
        r, s = mh.sort_basis_tuple(temp, self.alg.basis_strs)
        return (tuple(list(r) + [t[-1]]), s)

    def init_basis(self, deg:int)->None:
        """Sets value of deg in basis_cache and adds to basis_dicts"""
        if deg in self.basis_cache:
            return None
        deg_subsets = []
        for A in combinations([str(A) for A in self.alg.m_basis], deg):
            for B in self.alg.basis_strs:
                deg_subsets.append(tuple(list(A) + [B]))
        self.basis_cache[deg] = {}
        wght_ct = {}
        for i in range(len(deg_subsets)):
            # count the number of elements of deg d and wght w
            # and set the deg, wght, and index of each elt in dwi_dicts
            A = deg_subsets[i]
            w = self.tuple_wght(A)
            if w not in wght_ct:
                j = 0
                wght_ct[w] = 1
            else:
                j = wght_ct[w]
                wght_ct[w] += 1
            self.dwi_dicts[A] = (deg, w, j)
        for A in deg_subsets:
            d, w, i = self.dwi_dicts[A]
            vec = sp.SparseMatrix(sp.zeros(wght_ct[w], 1))
            vec[i] = 1
            vd = {d: {w: vec}}
            b_elt = CochainBasisElt(self, d, w, vd, A)
            if i == 0:
                self.basis_cache[deg][w] = [b_elt]
            else:
                self.basis_cache[deg][w].append(b_elt)

    def cb_mat(self, d:int, w:int)->sp.MatrixBase:
        """Returns the coboundary matrix which operates on C^d_w
        INPUTS:
        * 'd' - degree
        * 'w' - weight
        """
        if (d,w) in self.cb_mat_cache:
            return self.cb_mat_cache[(d,w)]
        if self.basis(d + 1, w) == []:
            self.cb_mat_cache[(d, w)] = sp.zeros(0, len(self.basis(d, w)))
        elif self.basis(d, w) == []:
            self.cb_mat_cache[(d, w)] = sp.zeros(len(self.basis(d + 1, w)), 0)
        else:
            self.cb_mat_cache[(d, w)] = sp.SparseMatrix(
                [list(c.cb_vec()) for c in self.basis(d, w)]
            ).transpose()
        return self.cb_mat_cache[(d, w)]
    
    def cd_mat(self, d:int, w:int)->sp.MatrixBase:
        """Returns the codifferential matrix which operates on C^d_w
        INPUTS:
        * 'd' - degree
        * 'w' - weight
        """
        return (self.Q(d,w).inverse_LU()*self.cb_mat(d-1,w)*self.Q(d-1,w)).transpose()
    
    def coordinatize_in_subspace(self,c:Cochain,d:int,w:int,subspace:str, 
                                 subsp_check:bool=False, check_result:bool=False)->sp.MatrixBase:
        """Returns the coordinatization of c in the basis for subspace.
        Checks that c is homogeneous degree d and weight w, but only checks
        that c is from the specified subspace if subsp_check is True
        
        INPUTS:
        * 'c' - a cochain from self homogeneous in degree and weight from subspace
        * 'd' - degree
        * 'w' - weight
        * 'subspace' - among 'closed', 'coclosed', 'exact', 'coexact', and 'harmonic'
        * 'subsp_check' - determine if c in subspace is checked
        * 'check_result' - determines if the result is checked"""
        
        # check homogeneity
        c.clear_zeros()
        for a in c.vd:
            if a!=d: raise Coordinatization_Exception(
                "Attempted to coorinatize cochain nonhomogeneous in degree")
            for b in c.vd[a]:
                if b!=w: raise Coordinatization_Exception(
                "Attempted to coorinatize cochain nonhomogeneous in weight")
        
        B=self.subspace_basis(subspace,d,w)
        if c.vd=={}: return sp.zeros(sp.shape(B)[1],1)
        if subsp_check:
            T=sp.Matrix([list(B.col(i)) for i in range(B.shape[1])]+[list(c.vd[d][w])])
            if T.rank()!=B.shape[1]: raise Coordinatization_Exception(
                "Attempted to coorinatize cochain not in specified subspace")
        
        M:sp.Matrix
        if (subspace,d,w) in self.pinv_cache: M=self.pinv_cache[(subspace,d,w)]
        else: M=self.subspace_basis(subspace,d,w).pinv()
        r=M*c.vd[d][w]

        if check_result:
            c_test=self.elt({d:{w:self.subspace_basis(subspace,d,w)*r}})-c
            ch.simplify_cochain(c_test)
            if c_test!=self.elt({}): print(
                'coordinatize_in_subspace failure')
        return r
        
    
    def cb_preim_mat(self,d:int,w:int)->sp.MatrixBase:
        """Returns the matrix which computes for each exact cochain in degree d 
        and weight w a preimage element from the coexact forms, using coords on
        the exact and coexact spaces. The matrix represents (d^*d)^{-1}*(d^*),
        which is an isomorphism from the exact forms to the coexact forms.
        INPUTS:
        * 'd' - degree
        * 'w' - weight
        """
        if (d,w) not in self.cb_preim_mat_cache: 
            self.cb_preim_mat_cache[(d,w)]=(
                (self.cd_mat(d,w)*self.cb_mat(d-1,w)).pinv()*self.cd_mat(d,w))
        return self.cb_preim_mat_cache[(d,w)]



    def cb(self, c:Cochain)->Cochain:
        """Returns the coboundary map of C(m,g) applied to c
        INPUTS:
        * 'c' - a cochain with self as parent"""
        return c.cb()

    def subspace_proj(self, c:Cochain, subspace:str)->Cochain:
        """Returns the projection of c onto subspace
        INPUTS:
        * 'subspace' - among 'closed', 'coclosed', 'exact', 'coexact', and 'harmonic'
        * 'c' - a cochain from self
        """
        r:dict[int,dict[int,sp.MatrixBase]] = {}
        for d in c.vd:
            r[d] = {}
            for w in c.vd[d]:
                r[d][w] = lh.ortho_proj(
                    c.vd[d][w], self.subspace_basis(subspace, d, w), self.Q(d, w)
                )
        return self.elt(r)

    def subspace_basis(self, subspace:str, d:int, w:int)->sp.MatrixBase:
        """Returns a matrix whose columns form a basis for the subspace 
        in degree d and weight w

        INPUTS:
        * 'subspace' - among 'closed', 'coclosed', 'exact', 'coexact', and 'harmonic'
        * 'd' - a degree
        * 'w' - a weight
        """
        if len(self.basis(d, w)) == 0:
            return sp.Matrix([])
        if (subspace, d, w) in self.subspace_cache:
            return self.subspace_cache[(subspace, d, w)]

        if subspace == "closed":
            col_list = self.cb_mat(d, w).nullspace()
        if subspace == "coclosed":
            col_list = lh.Mat_adjoint(
                self.cb_mat(d - 1, w), self.Q(d - 1, w), self.Q(d, w)
            ).nullspace()
        if subspace == "exact":
            col_list = self.cb_mat(d - 1, w).columnspace()
        if subspace == "coexact":
            col_list = lh.Mat_adjoint(
                self.cb_mat(d, w), self.Q(d, w), self.Q(d + 1, w)
            ).columnspace()
        if subspace == "harmonic":
            N = (self.cb_mat(d, w) * self.subspace_basis("coclosed", d, w)).nullspace()
            col_list = [self.subspace_basis("coclosed", d, w) * v for v in N]
        if len(col_list) == 0:
            self.subspace_cache[(subspace, d, w)] = sp.zeros(len(self.basis(d, w)), 0)
        else:
            self.subspace_cache[(subspace, d, w)] = sp.Matrix(
                [list(A) for A in col_list]
            ).transpose()
        return self.subspace_cache[(subspace, d, w)]

    def cb_preim_elt(self, c:Cochain,check=False)->Cochain|None:
        """Returns a cochain which maps to c under the coboundary.
        If c is not exact, returns None.
        INPUTS:
        * 'c' - an exact cochain
        """
        r:dict[int,dict[int,sp.MatrixBase]] = {}
        for d in c.vd:
            for w in c.vd[d]:
                # To do: Figure out why the exponentiation takes longer for the second version
                # t = lh.new_Mat_preim_elt(self.cb_mat(d - 1, w), c.vd[d][w])
                t = self.cb_preim_mat(d,w)*c.vd[d][w]
                if t is None:
                    return None
                if d - 1 not in r:
                    r[d - 1] = {}
                if w not in r[d - 1]:
                    r[d - 1][w] = t
                else:
                    r[d - 1][w] = r[d - 1][w] + t
        result=self.elt(r)
        if check:
            test=result.cb()-c
            ch.simplify_cochain(test)
            if test!=self.elt({}):
                raise Failed_Check_Exception("Failed check in cb_preim_elt")
        return self.elt(r)

    def curv_dict_to_cochain(self, c:dict[tuple[int,int],sp.MatrixBase]):
        """Returns a cochain from self representing the structure function or curvature c
        INPUTS:
        * 'c' -- a dictionary repping a structure function {(i,j): vec rep of [Xi,Xj]}
        """
        r:dict[tuple[str,...],sp.Expr] = {}
        for i in range(
            len(self.alg.basis) - len(self.alg.m_basis), len(self.alg.basis)
        ):
            for j in range(i + 1, len(self.alg.basis)):
                for k in range(len(self.alg.basis)):
                    if c[(i, j)][k] != 0:
                        r[
                            (
                                self.alg.basis_strs[i],
                                self.alg.basis_strs[j],
                                self.alg.basis_strs[k],
                            )
                        ] = c[(i, j)][k]
        return self.elt_from_cd(r)

    def curv_cochain_to_dict(self, c:Cochain)->dict[tuple[int,int],sp.MatrixBase]:
        """Returns a dict of form {(i,j): v such that F*v=[Fi,Fj]-[Xi,Xj]} representing the cochain
        * 'c' -- a dictionary repping a structure function {(i,j): vec rep of [Xi,Xj]}
        """
        r = {}
        for i in range(
            len(self.alg.basis) - len(self.alg.m_basis), len(self.alg.basis)
        ):
            for j in range(i + 1, len(self.alg.basis)):
                r[(i, j)] = sp.zeros(len(self.alg.basis), 1)

        if 2 not in c.vd:
            return r
        for w in c.vd[2]:
            v = c.vd[2][w]
            for t in range(len(c.vd[2][w])):
                if v[t] != 0:
                    i, j, k = [
                        self.alg.basis.index(A) for A in self.basis(2, w)[t].components
                    ]
                    r[(i, j)][k] = v[t]
        return r

    def one_cochain_mat_rep(self, c:Cochain)->sp.MatrixBase:
        """Returns a matrix representation of c as an element of Hom(g,g)
        INPUTS:
        * 'c' - a 1 cochain from self
        """
        if 1 not in c.vd:
            return sp.zeros(len(self.alg.basis))
        r = sp.zeros(len(self.alg.basis))
        for w in c.vd[1]:
            for a in range(len(c.vd[1][w])):
                i, j = [self.alg.basis.index(A) for A in self.basis(1, w)[a].components]
                r[j, i] = c.vd[1][w][a]
        return r

    def regular_normal_2_cochain(self, str_rep:str)->Cochain:
        """Constructs an arbitrary regular and normal (that is, positive and coclosed) 
        2-cochain with coefficients an indexed object from the IndexedBase str_rep"""
        g = self.alg
        if self.rnc is None:
            eta:Cochain = self.elt({})
            # # Construct a regular normal 2-cochain
            for i in range(3, len(g.basis)):
                for j in range(i + 1, len(g.basis)):
                    for k in range(len(g.basis)):
                        if g.basis[k].wght - g.basis[i].wght - g.basis[j].wght > 0:
                            t = (g.basis_strs[i], g.basis_strs[j], g.basis_strs[k])
                            eta = eta + self.elt_from_cd(
                                {t: sp.IndexedBase("eta")[i, j, k]}
                            )

            c_eta = self.subspace_proj(eta, "exact")

            norm_eqs = []
            for d in c_eta.vd:
                for w in c_eta.vd[d]:
                    for i in range(c_eta.vd[d][w].shape[0]):
                        if c_eta.vd[d][w][i, 0] != 0:
                            norm_eqs.append(c_eta.vd[d][w][i, 0])

            norm_sols = sp.solve(norm_eqs, check=False, simplify=False)

            eta = eta.subs(norm_sols)
            ch.simplify_cochain(eta)
            self.rnc = eta
        return self.rnc.subs({sp.IndexedBase("eta"): sp.IndexedBase(str_rep)})


def homog_Gerst_prod(f:Cochain, h:Cochain)->Cochain:
    """returns f circ h
    args:
      * 'f', 'h' - deg homogeneous elements from a cochain complex
    """
    if f == f.parent.elt({}) or h == h.parent.elt({}):
        return f.parent.alg.elt()
    fd = list(f.vd.keys())[0]
    hd = list(h.vd.keys())[0]

    r = f.parent.elt({})
    for w in f.parent.alg.ext_alg.basis(fd + hd - 1):
        for e_elt in f.parent.alg.ext_alg.basis(fd + hd - 1)[w]:
            t = Apply_Gerst_prod_on_base(f, h, e_elt)
            for i in range(len(t.vec)):
                if t.vec[i] != 0:
                    s = tuple(
                        [str(A) for A in e_elt.components] + [t.parent.basis_strs[i]]
                    )
                    r += f.parent.elt_from_cd({s: t.vec[i]})
    return r


def homog_Gerst_bracket(f:Cochain, h:Cochain)->Cochain:
    if f == f.parent.elt({}) or h == h.parent.elt({}):
        return f.parent.alg.elt()
    fd = list(f.vd.keys())[0]
    hd = list(h.vd.keys())[0]

    r = f.parent.elt({})
    for w in f.parent.alg.ext_alg.basis(fd + hd - 1):
        for e_elt in f.parent.alg.ext_alg.basis(fd + hd - 1)[w]:
            t = Apply_Gerst_prod_on_base(f, h, e_elt) - (-1) ** (
                fd + hd
            ) * Apply_Gerst_prod_on_base(h, f, e_elt)
            for i in range(len(t.vec)):
                if t.vec[i] != 0:
                    s = tuple(
                        [str(A) for A in e_elt.components] + [t.parent.basis_strs[i]]
                    )
                    r += f.parent.elt_from_cd({s: t.vec[i]})
    return r


def Gerst_prod(f:Cochain, h:Cochain)->Cochain:
    """returns f circ h"""
    C = f.parent
    r = C.elt({})

    f_degs = list(f.vd.keys())
    h_degs = list(h.vd.keys())

    for fd in f_degs:
        fp = f.deg_proj(fd)
        for hd in h_degs:
            hp = h.deg_proj(hd)
            r = r + homog_Gerst_prod(fp, hp)
    return r


def Gerst_bracket(f:Cochain, h:Cochain)->Cochain:
    C = f.parent
    r = C.elt({})

    f_degs = list(f.vd.keys())
    h_degs = list(h.vd.keys())

    for fd in f_degs:
        fp = f.deg_proj(fd)
        for hd in h_degs:
            hp = h.deg_proj(hd)
            r = r + homog_Gerst_bracket(fp, hp)
    return r
