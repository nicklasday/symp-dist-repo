from __future__ import annotations
__all__ = ["BianchiDict"]
import copy
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
        self.syz_ds_dict:dict[sp.Indexed,dict[tuple,sp.Expr]]={}
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

    def check_Bianchi(self,w:int,verbose=False):
        """Checks the Bianchi identity in weight w for self, returning the indices of the nonzero components.
        If verbose is set to True, then the nonzero components are printed.
        
        * 'w' - integer weight
        * 'verbose' - bool"""
        for t in self.tuples_by_wght[w]:
            nonzero_list=[]
            i,j,k,m=t
            if (i,j,k) in self.CartanGeometry.Bianchi_cache:
                zero_elt=self.CartanGeometry.Bianchi_cache[(i,j,k)]
            else:
                zero_elt=self.CartanGeometry.Bianchi(i,j,k)
                self.CartanGeometry.Bianchi_cache[(i,j,k)]=zero_elt
            r=sp.simplify(dsh.ds_subs(zero_elt.vec[m],self.ds_dict)[0])
            if r!=0: 
                if verbose: print(r)
                nonzero_list.append((i,j,k,m))
            return r,

    def add_key(self,
            key: sp.Indexed,
        val: sp.Expr,
        rel_Bianchi_keys: set[tuple] | None = None,
    ):
        dsh.ds_add_key(key,val,self.ds_dict,rel_Bianchi_keys,self.rel_Bianchi_dict,self.not_added)

    def compute_Bianchi(self,w,verbose=True):
        """Computes the Bianchi identity in weight w, updating self.ds_dict, and adding unused relations to self.not_added
        
        * 'w' - integer weight
        * 'verbose' - boolean"""
        for t in self.tuples_by_wght[w]:
            i,j,k,m=t
            time0=time.time()
            if verbose: print('Computing', t)
            if (i,j,k) in self.CartanGeometry.Bianchi_cache:
                zero_elt=self.CartanGeometry.Bianchi_cache[(i,j,k)]
            else: 
                time1=time.time()
                zero_elt=self.CartanGeometry.Bianchi(i,j,k)
                self.CartanGeometry.Bianchi_cache[(i,j,k)]=zero_elt
                if verbose: print('    Bianchi computed in',mh.hrs_min_sec(time.time()-time1))
            time2=time.time()
            to_solve,rel_keys=dsh.ds_subs(zero_elt.vec[m],self.ds_dict)
            rel_Bianchi_keys:set[tuple]=set()
            for a in rel_keys: rel_Bianchi_keys=rel_Bianchi_keys.union(self.rel_Bianchi_dict[a[0]][a[1]])
            rel_Bianchi_keys.add((i,j,k,m))

            if verbose: print('    to_solve computed in',mh.hrs_min_sec(time.time()-time2))
            if sp.simplify(to_solve)!=0:
                time3=time.time()
                s=mh.find_a_linear_term(to_solve,self.CartanGeometry.fund_invars)
                if s is None: s=mh.find_a_linear_term(to_solve)
                if s is None:
                    if verbose: print('no linear term in',t)
                    self.not_added.append(t)
                else:
                    sol=sp.solve(to_solve,s,dict=True)[0]
                    if verbose: print('    Solving complete in',mh.hrs_min_sec(time.time()-time3))
                    time4=time.time()
                    for a in sol: 
                        dsh.ds_add_key(a,sol[a],self.ds_dict,self.CartanGeometry.symbol,
                                       rel_Bianchi_keys,self.rel_Bianchi_dict,self.not_added)
                    if verbose: print('    Substitution complete in',mh.hrs_min_sec(time.time()-time4))
            if verbose: print('   ',t,'computed in',mh.hrs_min_sec(time.time()-time0))
            # Notice that D.curv = -P.curvature, since I switched sign conventions

    def subs_needed(self):
        """Returns True if the values of the differential substitution dictionary represented by self
        involve the keys, so that additional back substitution is possible. Returns False otherwise."""
        return dsh.ds_subs_needed(self.ds_dict)
    
    def reset_syzygies(self):
        """Recomputes self.syz_ds_dict from self.ds_dict, resetting any added assumptions 
        or other fiddling."""
        self.syz_ds_dict={}
        for A in self.CartanGeometry.fund_invars:
            if A in self.ds_dict: self.syz_ds_dict[A]=copy.deepcopy(self.ds_dict[A])
        return None
    
    def syzygies(self)->dict[int,list[sp.Expr]]:
        """Returns the syzygies in self, as a dictionary sorted by weight. This method
        assumes self.CartanGeometry.fund_invars is a correct list of the fundamental 
        invariants of the Cartan geometry."""
        
        if self.syz_ds_dict is None: self.reset_syzygies()
        temp_syz:dict[int,list[sp.Expr]]={}
        for k in self.syz_ds_dict:
            for j in self.syz_ds_dict[k]:
                w=-mh.wght_of_ind(k.base[k.indices+j],self.CartanGeometry.symbol)
                if w not in temp_syz: temp_syz[w]=[]
                new_syzygy:sp.Expr=sp.simplify(k.base[k.indices+j]-self.syz_ds_dict[k][j])
                if new_syzygy!=0:
                    new_syzygy=new_syzygy*new_syzygy.as_numer_denom()[1]
                    temp_syz[w].append(new_syzygy)

        r:dict[int,sp.Expr]={}
        ds_dict_w:dict[sp.Indexed,dict[tuple,sp.Expr]]={} # to weight at most w
        m=max(set(temp_syz.keys()).union(set([1])))
        for w in range(m):
            r[w+1]=[]
            for k in self.syz_ds_dict:
                for j in self.syz_ds_dict[k]:
                    if -mh.wght_of_ind(k.base[k.indices+j],self.CartanGeometry.symbol)<=w:
                        if k not in ds_dict_w: ds_dict_w[k]={}
                        ds_dict_w[k][j]=self.syz_ds_dict[k][j]
            
            if w+1 in temp_syz:
                for syz in temp_syz[w+1]:
                    s=sp.simplify(dsh.ds_subs(syz,ds_dict_w)[0])
                    if s!=0: r[w+1].append(s)
        for k in list(r.keys()):
            if r[k]==[]: del r[k]
        return r
    
    # I've adjusted this to modify ds_dict, rather than syzygy dict;
    # that will allow for the syzygy computations to use them from the beginning.
    # This turns out not to help much, so I'll probably just switch it back.
    def add_syzygy_assumption(self,assum:list[sp.Expr]):
        """Adds the assumptions in assum into self.syz_ds_dict. Relations appearing with 
        no linear term appear will be added to self.not_added"""
        for a in assum:
            dsh.add_expr_to_ds_dict(dsh.ds_subs(a,self.ds_dict)[0],
                                    self.ds_dict,
                                    self.CartanGeometry.symbol,
                                    rel_Bianchi_terms=set(),
                                    fund_invars=self.CartanGeometry.fund_invars,
                                    rel_Bianchi_dict=self.rel_Bianchi_dict,
                                    not_added=self.not_added)
            
    def check_fund_invars(self,w:int,s:str):
        """This method returns True if all structure functions s[i,j,k,l_1,l_2,...,l_m]
        are represented in terms of the fundamental invariants self.CartanGeometry.fund_invars
        using the substitutions from self, False otherwise.
        """
        # To do
        NotImplemented

    