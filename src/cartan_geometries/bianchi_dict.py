__all__ = ["BianchiDict"]
from __future__ import annotations
import time
from typing import TYPE_CHECKING

import sympy as sp
from ..utils import ds_helpers as dsh
from ..utils import math_helpers as mh

if TYPE_CHECKING:
    from cartan_geometry import RegularCartanGeometry

class BianchiDict(object):
    def __init__(self, P:RegularCartanGeometry,ds_dict:dict[sp.Indexed,dict[tuple,sp.Expr]]):
        self.CartanGeometry:RegularCartanGeometry=P
        self.ds_dict:dict[sp.Indexed,dict[tuple,sp.Expr]]=ds_dict
        self.not_added:list[sp.Expr]=[]
        self.rel_Bianchi_dict:dict[sp.Indexed,dict[tuple,sp.Expr]]={}
        self.tuples_by_wght:dict[int,list[tuple]]={}
        for i in range(3,len(P.symbol.basis)):
            for j in range(i+1,len(P.symbol.basis)):
                for k in range(j+1,len(P.symbol.basis)):
                    for m in range(len(P.symbol.basis)):
                        w=-P.symbol.basis[i].wght-P.symbol.basis[j].wght-P.symbol.basis[k].wght+P.symbol.basis[m].wght
                        if w not in self.tuples_by_wght: self.tuples_by_wght[w]=[]
                        self.tuples_by_wght[w].append((i,j,k,m))

    def check_Bianchi(self,w):
        for t in self.tuples_by_wght[w]:
            i,j,k,m=t
            if (i,j,k) in self.CartanGeometry.Bianchi_cache:
                zero_elt=self.CartanGeometry.Bianchi_cache[(i,j,k)]
            else:
                zero_elt=self.CartanGeometry.Bianchi(self.CartanGeometry,i,j,k)
                self.CartanGeometry.Bianchi_cache[(i,j,k)]=zero_elt
            r=sp.simplify(dsh.ds_subs(zero_elt.vec[m],self.ds_dict)[0])
            if r!=0: print(r)

    def add_key(self,
            key: sp.Indexed,
        val: sp.Expr,
        rel_Bianchi_keys: Any | None = None,
    ):
        dsh.ds_add_key(key,val,self.ds_dict,rel_Bianchi_keys,self.rel_Bianchi_dict,self.not_added)

    def compute_Bianchi(self,w):
        for t in self.tuples_by_wght[w]:
            i,j,k,m=t
            time0=time.time()
            print('Computing', t)
            if (i,j,k) in self.CartanGeometry.Bianchi_cache:
                zero_elt=self.CartanGeometry.Bianchi_cache[(i,j,k)]
            else: 
                time1=time.time()
                zero_elt=self.CartanGeometry.Bianchi(i,j,k)
                self.CartanGeometry.Bianchi_cache[(i,j,k)]=zero_elt
                print('    Bianchi computed in',mh.hrs_min_sec(time.time()-time1))
            time2=time.time()
            to_solve,rel_keys=dsh.ds_subs(zero_elt.vec[m],self)
            rel_Bianchi_keys:set[tuple]=set()
            for a in rel_keys: rel_Bianchi_keys=rel_Bianchi_keys.union(self.rel_Bianchi_dict[a[0]][a[1]])
            rel_Bianchi_keys.add((i,j,k,m))

            print('    to_solve computed in',mh.hrs_min_sec(time.time()-time2))
            if sp.simplify(to_solve)!=0:
                time3=time.time()
                s=mh.find_a_linear_term(to_solve,self.CartanGeometry.fund_invars)
                if s is None: s=mh.find_a_linear_term(to_solve)
                if s is None:
                    print('no linear term in',t)
                    self.not_added.append(t)
                    self.rel_Bianchi_dict[t]=rel_Bianchi_keys
                else:
                    sol=sp.solve(to_solve,s,dict=True)[0]
                    print('    Solving complete in',mh.hrs_min_sec(time.time()-time3))
                    time4=time.time()
                    for a in sol: 
                        dsh.ds_add_key(a,sol[a],self.ds_dict,D,rel_Bianchi_keys,self.rel_Bianchi_dict,self.not_added)
                    print('    Substitution complete in',mh.hrs_min_sec(time.time()-time4))
            print('   ',t,'computed in',mh.hrs_min_sec(time.time()-time0))
            # Notice that D.curv = -P.curvature, since I switched sign conventions