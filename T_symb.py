from sympy import *

class T_symb_elt:
    
    def __init__(self,vec,parent):
        '''vec: a list of length len(self.parent.basis) with integer entries'''
        self.parent=parent
        self.vec=Matrix(vec)
        if shape(self.vec)[1]!=1: self.vec=self.vec.transpose() # column vector
        if shape(self.vec)[0]!=parent.heis_dim+4:
            raise constr_Mat_size_exception
        self.heis_dim=parent.heis_dim
        self.ad_mat_cache={}
        
    def __str__(self):
        if self==0: return '0'
        result=''
        cntr=0
        while result=='':
            if self.vec[cntr]!=0:
                if self.vec[cntr]==1:
                    result=str(self.parent.basis[cntr])
                elif self.vec[cntr]==-1:
                    result='-'+str(self.parent.basis[cntr])
                elif type(self.vec[cntr])==Add:
                    result='('+str(self.vec[cntr])+')*'+str(self.parent.basis[cntr])
                else:
                    result = str(self.vec[cntr])+'*'+str(self.parent.basis[cntr])
            cntr+=1
        for i in range(cntr,len(self.parent.basis)):
            if self.vec[i]==1:
                result+=' + '+str(self.parent.basis[i])
            elif self.vec[i]==-1:
                result+=' - '+str(self.parent.basis[i])
            elif self.vec[i]!=0:
                if type(self.vec[i])==Add: c_str='('+str(self.vec[i])+')*'
                else: c_str=str(self.vec[i])
                result+=' + '+c_str+'*'+str(self.parent.basis[i])
        return result

    def __repr__(self):
        return str(self)
    
    def __eq__(self,other):
        if other==0 and type(other)==int:
            return self.vec==Matrix([0]*(len(self.parent.basis)))
        if not hasattr(other,'parent'): return False
        if self.parent!=other.parent: return False
        return self.vec==other.vec
    
    def __neg__(self):
        return(T_symb_elt([-A for A in self.vec],self.parent))
    
    def __add__(self,other):
        if other==0:
            return self
        return T_symb_elt(self.vec+other.vec,self.parent)   
    
    def __radd__(self,other):
        return self+other
    
    def __sub__(self,other):
        return self+(-other)
            
    def __mul__(self,other):
        return T_symb_elt([other*A for A in self.vec],self.parent)
    
    def __rmul__(self,other):
        return self*other
        
    def ad_mat(self,mod='g',d=None):
        """returns the matrix representing the ad action of self on a module among
        g (default), m, g^*, exterior alg of m, or CE complex of m with coeffs in g.
        
        INPUT:
            * "mod" -- among 'g','m','g_dual','m_dual','E', and 'CE'"""
        if mod not in self.ad_mat_cache:
            r=SparseMatrix(zeros(len(self.parent.basis)))
            for i in range(len(self.parent.basis)):
                if self.vec[i]!=0: 
                    r+=self.vec[i]*self.parent.ad_mats(i,mod,d)
            self.ad_mat_cache[mod]=r
        return self.ad_mat_cache[mod]
        
    def ad(self,t,dual=False):
        '''returns: a T_symb_elt object representing ad(self,t)
           if dual=True, the object returned represents ad(self,t^*)'''
        P=self.parent
        E=self.parent.ext_alg
        C=self.parent.cochain_complex
        
        # # g and g_dual
        if type(t)==T_symb_elt or type(t)==T_symb_basis_elt:
            if t.parent!=P: raise invalid_parent_exception
            if dual: mod='g_dual'
            else: mod='g'
            if self==0 or t==0:
                return P.elt([0]*len(P.basis))
            return P.elt(self.ad_mat(mod)*t.vec)
  
        # # To Do: Implement ad for exterior algebra and cochains
        # # (Replace deprecated code below)
       
        ### Begin Deprecated ------------------------------------------------
        if type(t)==ext_elt:
            if t.parent!=E: raise invalid_parent_exception
            if t==0 or self==0: return E.elt({})
        
            # result=E.elt()
            # for E in 
            
            result=E.elt({})
            for k in t.coeff_dict:
                deg=E.deg(E.elt({k:1}))
                for i in range(len(P.basis)):
                    if self.vec[i]!=0:
                        result+=self.vec[i]*t.coeff_dict[k]*P.basis[i].ext_ad_dicts[deg][k]
            return result
        
        if type(t)==cochain:
            if t.parent!=C: raise invalid_parent_exception
            if t==0 or self==0: return C.elt({})
            
            result=C.cochain({})
            for k in t.coeff_dict:
                deg=C.deg(C.cochain({k:1}))
                for i in range(len(P.basis)):
                    if self.vec[i]!=0:
                        result+=self.vec[i]*t.coeff_dict[k]*P.basis[i].cochain_ad_dicts[deg][k]
            return result
        raise ValueError('ad must only be applied to objects T_symb_elt, T_symb_basis_elt, ext_elt, and cochain')
        ### End Deprecated ------------------------------------------------
        
    def iprod(self,other):
        '''other: another T_symb_elt or T_symb_basis_elt
           returns: the inner product of self and other'''
        if not hasattr(other,'parent'): raise ValueError('iprod recieved an invalid arg') 
        if self.parent!=other.parent: raise invalid_parent_exception
        return((self.vec.transpose()*self.parent.Q*other.vec)[0])
    
    def cast_as_ext_elt(self):
        result_dict=remove_zeros({(self.parent.basis_strs[i],):self.vec[i] for i in range(len(self.vec))})
        return self.parent.ext_alg.elt(result_dict)
    
    def cast_as_cochain(self):
        result_dict=remove_zeros({(self.parent.basis_strs[i],):self.vec[i] for i in range(len(self.vec))})
        return self.parent.cochain_complex.cochain(result_dict)
    
    
    
class T_symb_basis_elt(T_symb_elt):
    def __init__(self,str_rep,wght,parent):
        self.str_rep=str_rep
        self.wght=wght
        tvec=SparseMatrix([0]*(parent.heis_dim+4))
        
        basis_str_list=['Y','H','E','X']
        for i in range(1,parent.heis_dim):
            basis_str_list.append('e%d'%i)
        basis_str_list.append('N')
        tvec[basis_str_list.index(str_rep)]=1
        
        super().__init__(tvec,parent)

        self.ad_dict={}
        
        ###  Begin Deprecated --------------------------------------------
        self.dual_ad_dict={}
        self.cochain_ad_dicts={}
        self.ext_ad_dicts={}
        ###  End Deprecated --------------------------------------------
        
    def __str__(self):
        return self.str_rep
    
    def __repr__(self):
        return self.str_rep
    
    def __lt__(self,other):
        if self.parent!=other.parent: raise invalid_parent_exception 
        return self.parent.basis.index(self)<self.parent.basis.index(other)
    
    def __gt__(self,other):
        if self.parent!=other.parent: raise invalid_parent_exception 
        return self.parent.basis.index(self)>self.parent.basis.index(other)
    
    def __le__(self,other):
        if self.parent!=other.parent: raise invalid_parent_exception 
        return self.parent.basis.index(self)<=self.parent.basis.index(other)

    def __ge__(self,other):
        if self.parent!=other.parent: raise invalid_parent_exception 
        return self.parent.basis.index(self)>=self.parent.basis.index(other)
    
    def ad_mat(self,mod='g',d=None):
        return self.parent.ad_mats(self.parent.basis.index(self),mod,d)



class T_symb:
    def __init__(self,heis_dim):
        if heis_dim%2!=1 or heis_dim<1: raise heis_dim_exception
           
        self.heis_dim=heis_dim
        self.basis_strs=['Y','H','E','X']
        for i in range(1,heis_dim):
            self.basis_strs.append('e%d'%i)
        self.basis_strs.append('N')
        
        # Weights
        self.wght_list=[1,0,0,-1]
        for i in range(4,len(self.basis_strs)-1):
            self.wght_list.append(-i+3)
        self.wght_list.append(-heis_dim)
        
        self.basis=[T_symb_basis_elt(self.basis_strs[i],self.wght_list[i],self) 
                    for i in range(len(self.basis_strs))]
        self.cochain_complex=cochain_complex(self)
        self.ext_alg=ext_alg(self)
        
        self.ad_dict={}
        self.ad_mat_cache={}
        self.set_ad_dict()
        
        #self.cochain_complex.init_ad_2_cochain()
        self.Q=SparseMatrix(diag(*[1,2,2,1]+[factorial(i-1)/factorial(heis_dim-i-1)
                                       for i in range(1,heis_dim)]+[1]))
        self.iprod_list=[1,2,2,1]+[factorial(i-1)/factorial(heis_dim-i-1) for i in range(1,heis_dim)]+[1]
        

        
    def set_ad_dict(self):
        # I'll try to avoid using this
        # Following Medvedev's basis/conventions, except the error in [Y,e_i]=(i-1)*(2*m+1-i)e_{i-1}
        k=len(self.basis)
        V_basis=self.basis[4:self.heis_dim+3]

        Y=self.basis[0]
        H=self.basis[1]
        E=self.basis[2]
        X=self.basis[3]
        N=self.basis[-1]
        ## Set ad_dicts
        #  First, set ad_dict for each T_symb_basis_elt

        for A in self.basis[0:4]:
            E.ad_dict[str(A)]=T_symb_elt([0]*k,self)
        for i in range(1,len(V_basis)+1):
            E.ad_dict[str(V_basis[i-1])]=V_basis[i-1]
        E.ad_dict['N']=2*N

        X.ad_dict['Y']=H
        X.ad_dict['H']=-2*X
        X.ad_dict['E']=T_symb_elt([0]*k,self)
        X.ad_dict['X']=T_symb_elt([0]*k,self)
        for i in range(1,len(V_basis)):
            X.ad_dict[str(V_basis[i-1])]=V_basis[i]
        X.ad_dict[str(V_basis[len(V_basis)-1])]=T_symb_elt([0]*k,self)
        X.ad_dict['N']=T_symb_elt([0]*k,self)

        Y.ad_dict['Y']=T_symb_elt([0]*k,self)
        Y.ad_dict['H']=2*Y
        Y.ad_dict['E']=T_symb_elt([0]*k,self)
        Y.ad_dict['X']=-H
        Y.ad_dict[str(V_basis[0])]=T_symb_elt([0]*k,self)
        for i in range(2,len(V_basis)+1):
            Y.ad_dict[str(V_basis[i-1])]=(i-1)*(self.heis_dim-i)*V_basis[i-2]
        Y.ad_dict['N']=T_symb_elt([0]*k,self)

        H.ad_dict['Y']=-2*Y
        H.ad_dict['H']=T_symb_elt([0]*k,self)
        H.ad_dict['E']=T_symb_elt([0]*k,self)
        H.ad_dict['X']=2*X
        for i in range(1,len(V_basis)+1):
            H.ad_dict[str(V_basis[i-1])]=(2*i-self.heis_dim)*V_basis[i-1]
        H.ad_dict['N']=T_symb_elt([0]*k,self)

        for i in range(1,len(V_basis)+1):
            if i>=2: V_basis[i-1].ad_dict['Y']=-(i-1)*(self.heis_dim-i)*V_basis[i-2]
            else: V_basis[i-1].ad_dict['Y']=T_symb_elt([0]*k,self)
            V_basis[i-1].ad_dict['H']=-(2*i-self.heis_dim)*V_basis[i-1]
            V_basis[i-1].ad_dict['E']=-V_basis[i-1]
            if i<=self.heis_dim-2: V_basis[i-1].ad_dict['X']=-V_basis[i]
            else: V_basis[i-1].ad_dict['X']=T_symb_elt([0]*k,self)
            for j in range(1, len(V_basis)+1):
                if i+j==self.heis_dim: V_basis[i-1].ad_dict[str(V_basis[j-1])]=(-1)**i*N
                else: V_basis[i-1].ad_dict[str(V_basis[j-1])]=T_symb_elt([0]*k,self)
            V_basis[i-1].ad_dict['N']=T_symb_elt([0]*k,self)

        N.ad_dict['Y']=T_symb_elt([0]*k,self)
        N.ad_dict['H']=T_symb_elt([0]*k,self)
        N.ad_dict['E']=-2*N
        N.ad_dict['X']=T_symb_elt([0]*k,self)
        for i in range(1,len(V_basis)+1):
            N.ad_dict[str(V_basis[i-1])]=T_symb_elt([0]*k,self)
        N.ad_dict['N']=T_symb_elt([0]*k,self)
        
    def ad_mats(self,i,mod='g',d=None):
        """Returns the adjoint matrix of self.basis[i] acting on mod
        INPUTS:
            * "i" -- integer index of basis element
            * "mod" -- module of action, among 'g', 'g_dual','m','m_dual,'Emd','CE'
            * "d" -- degree of exterior/cochain, if applicable
            
        Importantly, m, m_dual, Emd (Exterior alg of m_dual), and CE=C(m,g) are only
        m-modules, not g-modules. Accordingly, g_+ basis elts return NONE for these modules
        """
        if (i,mod) in self.ad_mat_cache: return self.ad_mat_cache[(i,mod)]
    
        # # g, g_dual
        if mod in ('g','g_dual'):
            r=SparseMatrix(zeros(len(self.basis)))
            for j in range(len(self.basis)):
                A=self.basis[i].ad_dict[str(self.basis[j])]
                set_col(r,j,A.vec.transpose())
            if mod=='g_dual': self.ad_mat_cache[(i,mod)]=-r.transpose()
            else: self.ad_mat_cache[(i,mod)]=r
            return self.ad_mat_cache[(i,mod)]
        
        # # m, m_dual
        if mod in ('m','m_dual'):
            if i<3: return None # these are only m modules, not g modules
            self.ad_mats(self,(i,'g')) # cache the matrices for (i,'g') and (i,'g_dual')
            r=self.ad_mat_cache(i,'g')
            self.ad_mat_cache[(i,'m')]=r[3:shape(r)[0],3:shape(r)[1]] # cache m
            rd=-self.ad_mat_cache(i,'g_dual').transpose()
            self.ad_mat_cache[(i,'m_dual')]=sd[3:shape(rd)[0],3:shape(rd)[1]] # cache m_dual
            return self.ad_mat_cache[(i,mod)]
        
        # To do: Implement ad_mats for 'E','CE'
    
    def set_ad_mats(self):
        for i in range(len(self.basis)):
            A=self.basis[i]
            r=SparseMatrix([A.ad(B).vec for B in self.basis])
            self.ad_mats.append(r.transpose())
            
    def Ad_mat(self,A):
        '''arg: a T_symb_elt or T_symb_basis_elt
           returns: the matrix rep of Ad(exp(A))'''
        ad_mat=SparseMatrix(zeros(len(self.basis)))
        for i in range(len(self.basis)):
            ad_mat+=A.vec[i]*self.ad_mats(i,'g')
        return exp(ad_mat)
            
    
    def jacobi_test(self):
        '''returns: True if the Jacobi identity holds, False otherwise'''
        for A in self.basis:
            for B in self.basis:
                for C in self.basis:
                    t1=self.ad(A,self.ad(B,C))
                    t2=self.ad(self.ad(A,B),C)+self.ad(B,self.ad(A,C))
                    if t1!=t2: return False
        return True   
    
    def elt(self,vec=None):
        if vec==None: return(T_symb_elt([0]*len(self.basis),self))
        return T_symb_elt(vec,self)
    
    def sort_basis_tuple(self,basis_tuple):
        '''basis_tuple: a tuple of str_reps of T_symb_basis_elt objs
           returns: a tuple containing an rearrangement of basis_tuple of descending degree, 
           and the sign of the permutation (either -1 or 1)'''
        basis_list=list(basis_tuple)
        sorted_list=basis_list.copy()
        sorted_list.sort(key=lambda A:self.basis_strs.index(A))
        return(tuple(sorted_list),permutation_sign(basis_list,sorted_list))
    
    
#     ### Begin Deprecated---------------------------------------------------
    
#     def set_ext_ad_dict(self,deg):
#         '''sets A.ext_ad_dicts[deg] for each A in self.basis'''
#         # Don't reset a dict that's already been written
#         if deg in self.basis[0].ext_ad_dicts: return
#         self.ext_alg.init_basis(deg)
        
#         # initialize if necessary
#         if self.ext_alg==None: self.ext_alg=ext_alg(self)
#         if deg not in self.ext_alg.basis:
#             self.ext_alg.init_basis(deg)
            
#         # construct the dict
#         for A in self.basis:
#             A.ext_ad_dicts[deg]={}
        
#         # set the dict
#         if deg==0:
#             for A in self.basis:
#                 # ad(A,1)=0
#                 A.ext_ad_dicts[0]={tuple():0}
#             return
        
#         # set the previous dict
#         #I'm not sure how to incorporate pickling into this call
#         self.set_ext_ad_dict(deg-1)
        
#         if deg==1:
#             for A in self.basis:
#                 for B in self.basis:
#                     ext_B=self.ext_alg.elt({(str(B),):1})
#                     temp=A.dual_ad_dict[str(B)].vec
#                     A.ext_ad_dicts[1][(str(B),)]=self.ext_alg.elt({(str(self.basis[i]),):temp[i]
#                                                         for i in range(len(self.basis))})
#             return
        
#         for A in self.basis:
#             for B in self.ext_alg.basis_strs[deg]:
#                 B1=B[0:len(B)-1]
#                 ext_B1=self.ext_alg.elt({B1:1})
#                 B2=B[len(B)-1:len(B)]
#                 ext_B2=self.ext_alg.elt({B2:1})
#                 A.ext_ad_dicts[deg][B]=(A.ext_ad_dicts[deg-1][B1].wedge(ext_B2)
#                                        +ext_B1.wedge(A.ext_ad_dicts[1][B2]))
                
#     # To Do: finish writing this algorithm
#     # (Copy pasted from the above)
#     def set_cochain_ad_dict(self,deg):
#         # Don't reset a dict that's already been written
#         if deg in self.basis[0].cochain_ad_dicts: return
        
#         # initialize if necessary
#         if self.cochain_complex==None: init_cochain_complex(self)
#         C=self.cochain_complex
#         if deg not in C.basis:
#             C.init_basis(deg)
            
#         # construct the dict
#         for A in self.basis:
#             A.cochain_ad_dicts[deg]={}
        
#         # set the dict
#         if deg==0:
#             for A in self.basis:
#                 for B in self.basis:
#                     v=A.ad_dict[str(B)].vec
#                     A.cochain_ad_dicts[0][(str(B),)]=C.cochain({(self.basis_strs[i],):v[i]
#                                                                 for i in range(len(self.basis))})
#             return
        
#         self.set_ext_ad_dict(deg)
#         for A in self.basis:
#             for B1 in self.ext_alg.basis_strs[deg]:
#                 for B2 in self.basis:
#                     k=B1+(str(B2),)
#                     c_B2=C.cochain({(str(B2),):1})
#                     v=A.ad_dict[str(B2)].vec
#                     c_ad_B2=C.cochain({(str(self.basis[i]),):v[i] for i in range(len(self.basis))})
#                     t1=self.ext_alg.elt({B1:1}).wedge(c_ad_B2)
#                     t2=A.ext_ad_dicts[deg][B1].wedge(c_B2)
#                     A.cochain_ad_dicts[deg][k]=t1+t2
                    
#     ### End Deprecated---------------------------------------------------
    
    def ad(self,t,c):
        '''t: a T_symb_elt object
           c: an object of type T_symb_basis_elt, T_symb_elt, ext_elt, or cochain
           returns: ad(t,c)'''
        if type(t)==T_symb_elt or type(t)==T_symb_basis_elt:
            return t.ad(c)
        
        if type(t)==int and t==0: # should I use 'is' here?
            if type(c)==cochain:
                return c.parent.cochain({})
            if type(c)==ext_elt:
                return c.parent.elt({})
            if type(c)==T_symb_elt or type(c)==T_symb_basis_elt:
                return c.parent.elt([0]*len(c.parent.basis))
        raise ValueError('The first argument of ad should be of type T_symb_elt or T_symb_basis_elt')
        
    def iprod(self,t1,t2):
        '''t1,t2: of the same type among T_symb_basis_elt, T_symb_elt, ext_elt, and cochain'''
        if not hasattr(t1,'parent') and hasattr(t2,'parent'):
            raise ValueException('arguments of iprod must have type T_symb_basis_elt, T_symb_elt, ext_elt, or cochain')
        
        if t1==0 or t2==0:
            return 0
        
        if t1.parent==t2.parent:
            return t1.iprod(t2)
        
        raise invalid_parent_exception('arguments of iprod must have the same parent')