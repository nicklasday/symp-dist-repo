from sympy import factorial

class T_symb_elt:
    def __init__(self,vec,parent):
        '''vec: a list of length len(self.parent.basis) with integer entries'''
        self.parent=parent
        self.vec=Matrix(vec)
        if shape(self.vec)[1]!=1: self.vec=self.vec.transpose() # column vector
        if shape(self.vec)[0]!=len(parent.basis_strs):
            raise constr_Mat_size_exception()
        self.ad_mat_cache={}
        self.ext_alg=ext_alg(self)
        self.cochain_complex=cochain_complex(self)
        # To Do: Introduce filtered base here
        
    def __str__(self):
        return str_from_vec(self.vec,self.parent.basis)

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
        
    def Ad_mat(self,mod='g',d=None,w=None,filtered=False):
        """Returns the Adjoint matrix of self acting on mod
        INPUTS:
            * "mod" -- module of action, among 'g', 'g_dual','m','m_dual,'E','CE'
            * "d" -- degree of exterior element/cochain, if applicable
            * "w" -- weight of exterior element/cochain, if applicable
            
        Importantly, m, m_dual, Emd (Exterior alg of m_dual), and CE=C(m,g) are only
        m-modules, not g-modules. Accordingly, g_+ basis elts return NONE for these modules
        
        If mod is among 'E' or 'CE' the result is a dictionary with keys (td,tw), target deg and wght
        """
        if mod in ['E','CE']:
            return NotImplemented
        if mod in ['g','g_dual','m','m_dual']:
            if d!=None or w!=None: raise NotImplementedException
            return exp(self.ad_mat(mod))

    def ad_mat(self,mod='g',d=None,w=None,filtered=False):
        """Returns the adjoint matrix of self acting on mod, or a dict of such matrices
        INPUTS:
            * "mod" -- module of action, among 'g', 'g_dual','m','m_dual,'E','CE'
            * "d" -- degree of exterior element/cochain, if applicable
            * "w" -- weight of exterior element/cochain, if applicable
            
        Importantly, m, m_dual, Emd (Exterior alg of m_dual), and CE=C(m,g) are only
        m-modules, not g-modules. Accordingly, g_+ basis elts return NONE for these modules
        
        If mod is among 'E' or 'CE' the result is a dictionary with keys (td,tw), target deg and wght
        """
        if mod in ['E','CE']:
            return NotImplmented
            # I shouldn't need this
        
        if mod in ['g','g_dual','m','m_dual']:
            if d!=None or w!=None: raise NotImplementedException
            r=SparseMatrix(zeros(self.parent.mod_dim(mod))) # a matrix of appropriate dim
            for i in range(len(self.parent.basis)): 
                if type(self.vec[i])==type(IndexedBase('K')[0]):
                    for j in range(shape(self.parent.basis[i].ad_mat(mod))[0]):
                        for k in range(shape(self.parent.basis[i].ad_mat(mod))[1]):
                            r[j,k]+=self.vec[i]*self.parent.basis[i].ad_mat(mod)[j,k]
                else: r+=self.vec[i]*self.parent.basis[i].ad_mat(mod)
            return r
        
    def ad(self,t,mod='g'):
        """Returns an object representing ad(self,t), where t has constant coefficients
           INPUTS:
           * 't' - an element of mod with constant coefficients (i.e., no symbols or indexed objects)
           * 'mod' - among 'g','m','g_dual','m_dual','E,', and 'C'
           """
        P=self.parent
        E=self.parent.ext_alg
        C=self.parent.cochain_complex
        
        if t.parent==E: mod='E'
        if t.parent==C: mod='CE'
        
        if mod in ['E','CE']:
            r={}
            for sd in t.vd: # source deg
                for sw in t.vd[sd]: # source wght
                    for i in range(len(P.basis)): # entries of self
                        if self.vec[i]!=0:
                            td=sd # target deg
                            tw=sw+P.basis[i].wght # target weight
                            im_vec=self.vec[i]*P.basis[i].ad_mat(mod,sd,sw)*t.vd[sd][sw]
                            if td not in r: r[td] = {}
                            if tw not in r[td]: r[td][tw]=zeros(len(t.parent.basis(td,tw)),1)
                            r[td][tw]+=im_vec
            if mod=='E': return E.elt(r)
            if mod=='CE': return C.elt(r)
                        
        if mod in ['g','g_dual']: return t.parent.elt(self.ad_mat(mod)*t.vec)
        if mod in ['m','m_dual']: 
            temp=self.ad_mat(mod)*Matrix(t.vec[3:len(t.vec)])
            return t.parent.elt([0]*3+list(temp))

    def __Ad_wght_0(self,t):
        P=self.parent
        C=P.cochain_complex
        r={}
        
        for d in t.vd: # source deg
            if d not in r: r[d] = {}
            for w in t.vd[d]: # source wght
                exp_list=[0]*len(t.vd[d][w])
                for i in range(len(P.basis)):
                    if P.basis[i].wght==0:
                        P.basis[i].init_ad_mat('CE',d,w)
                        for j in range(len(exp_list)):
                            exp_list[j]+=self.vec[i]*P.basis[i].scalar_cache['CE'][d][w][j]
                coeff_list=[exp(A) for A in exp_list]
                r[d][w]=Matrix([[t.vd[d][w][i]*coeff_list[i]] for i in range(len(coeff_list))])
        return self.parent.cochain_complex.elt(r)

    def __Ad_wght_1(self,t,mod='CE'):
        P=self.parent
        C=self.parent.cochain_complex
        r={}
        # scrub t of shape (0,0) entries
        for d in t.vd:
            for w in list(t.vd[d].keys()):
                if shape(t.vd[d][w])[1]==0:t.vd[d].pop(w)
        
        for d in t.vd:
            r[d]={}
            for w in t.vd[d]:
                r[d][w]=t.vd[d][w]
                if shape(r[d][w])[0]==0 or shape(r[d][w])[1]==0:
                    print('deg',d,'wght',w,'has shape', shape(r[d][w]))
        for d in t.vd: # source deg
            for w in t.vd[d]: # source wght
                for i in range(len(P.basis)): # entries of self
                    if self.vec[i]!=0 and P.basis[i].wght==1:
                        dom_vec=t.vd[d][w]
                        sw,tw=(w,w+P.basis[i].wght) # source and target deg and wght
                        if d not in r: r[d] = {}
                        k=1
                        while dom_vec!=zeros(*shape(dom_vec)):
                            im_vec=(self.vec[i]*P.basis[i].ad_mat(mod,d,sw)*dom_vec)/k
                            if tw not in r[d]: 
                                r[d][tw]=zeros(len(C.basis(d,tw)),1)
                            r[d][tw]+=im_vec
                            
                            dom_vec=im_vec
                            sw,tw=(sw+P.basis[i].wght,tw+P.basis[i].wght)
                            k+=1
        return C.elt(r)
        
    def Ad(self,t,mod='g',coord='second kind'):
        """For the cochain complex, this returns an object representing Ad(exp(w0)exp(w1),t) if coord=='second kind', 
        where t has constant coefficients and w0+w1=self is the weight decomposition of self.
        If coord=='first kind', this returns exp(ad(self))
        
           INPUTS:
           * 't' - an element of mod with constant coefficients
           * 'mod' - among 'g','m','g_dual','m_dual','E,', and 'C'
           * 'coord' - among 'first kind' and 'second kind'
           """
        P=self.parent
        E=self.parent.ext_alg
        C=self.parent.cochain_complex
        
        if t.parent==E: mod='E'
        if t.parent==C: mod='CE'

        if mod=='CE':
            if coord=='first kind': return NotImplemented # I shouldn't need this, really
            return self.__Ad_wght_0(self.__Ad_wght_1(t))
        if mod in ['g','g_dual','m','m_dual']:
            if coord=='first kind': return t.parent.elt(self.Ad_mat(mod)*t.vec)
            if coord=='second kind': 
                return t.parent.elt(exp(P.second_kind_to_first(self.vec).ad_mat(mod))*t.vec)
        
    def iprod(self,other):
        """Returns the inner product of self and other
        INPUTS:
        * 'other' - another T_symb_elt or T_symb_basis_elt
        """
        return self.parent.iprod(self,other)
    
    def cast_as_ext_elt(self):
        l=len(self.parent.basis)-len(self.parent.m_basis)
        if self.vec[0:l]!= [0]*l: print('T_symb_elt with positive component cannot be cast as ext_elt')
        return self.parent.ext_alg.elt_from_cd({(self.parent.basis_strs[i],):self.vec[i]
                                            for i in range(len(self.parent.basis)) if self.vec[i]!=0})
    
    def cast_as_cochain(self):
        return self.parent.cochain_complex.elt_from_cd({(self.parent.basis_strs[i],):self.vec[i]
                                            for i in range(len(self.parent.basis)) if self.vec[i]!=0})

# %% [markdown]
# A short remark on dual modules: $\mathfrak{m}^*$ is separately a $\mathfrak{g}^0$-module via $X\mapsto -\big(\text{ad}X|_{\mathfrak{m}}\big)^*$ and a $\mathfrak{g}_+$-module via $X\mapsto-\big(\text{ad}X\big)^*|_{\mathfrak{m}^*}$.
# 
# We want $\mathfrak{g}_+$-action for the purposes of the Ad action on the cochain complex

# %%
class T_symb_basis_elt(T_symb_elt):
    def __init__(self,str_rep,wght,parent,i,am):
        self.str_rep=str_rep
        self.wght=wght
        tvec=SparseMatrix([0]*(len(parent.basis_strs)))
        tvec[i]=1
        
        super().__init__(tvec,parent)
        self.ad_mat_cache={'g':SparseMatrix(am)}
        self.length=None
        if parent.Q!=None:
            i=parent.basis_strs.index(str_rep)
            self.length=parent.Q[i,i]
        self.scalar_cache={'CE':{},'E':{}}
        
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
    
    def ad_mat(self,mod='g',deg=None,wght=None):
        """Returns the mat rep of ad(self) on mod, either as a matrix
        or as a dictionary with keys (target_deg, target_wght) and matrix values
        """
        if mod=='g': return self.ad_mat_cache['g'] # cached upon initialization
        if mod=='g_dual': return -self.ad_mat('g').transpose()
        if mod=='m':
            r=self.ad_mat('g')
            return r[3:shape(r)[0],3:shape(r)[1]]
        if mod=='m_dual': 
            return -self.ad_mat('m').transpose()
        
        if deg==None or wght==None: print('ad mat requires deg and wght args for tensor modules')
        if (mod,deg,wght) not in self.ad_mat_cache: self.init_ad_mat(mod,deg,wght)
        return self.ad_mat_cache[(mod,deg,wght)]
    
    def init_ad_mat(self,mod='g',deg=None,wght=None):
        """caches the adjoint matrix of this element acting on mod
        INPUTS:
            * "mod" -- module of action, among 'g', 'g_dual','m','m_dual,'E','CE'
            * "deg" -- degree of exterior element/cochain, if applicable
            * "wght" -- wght of exterior element/cochain, if applicable
            
        Importantly, m, m_dual, Emd (Exterior alg of m_dual), and CE=C(m,g) are only
        m-modules, not g-modules. Accordingly, g_+ basis elts return NONE for these modules
        """
        
        if mod=='E':
            if ('E',deg,wght) in self.ad_mat_cache: return None
            E=self.parent.ext_alg
            col_list=[]

            ## Deal with len(basis)=0 cases first
            l_dom=len(self.parent.ext_alg.basis(deg,wght))
            l_codom=len(self.parent.ext_alg.basis(deg,wght+self.wght))
            if l_dom==0 or l_codom==0:
                self.ad_mat_cache[('E',deg,wght)]=SparseMatrix(zeros(l_codom,l_dom))
                return self.ad_mat_cache[('E',deg,wght)]

            for k in range(len(E.basis(deg,wght))):
                W=E.basis(deg,wght)[k] 
                comp=W.components
                # compute ad(self)(W) as a vector
                v=zeros(len(E.basis(deg,wght+self.wght)),1)
                for i in range(deg): 
                    adXi=self.ad_mat(mod='m_dual').col(self.parent.m_basis.index(comp[i]))
                    for j in range(len(self.parent.m_basis)):
                        if adXi[j]!=0:
                            s=[str(A) for A in comp]
                            t=s[0:i]+s[i+1:len(s)]
                            if self.parent.m_basis_strs[j] not in t:
                                s[i]=self.parent.m_basis_strs[j]
                                s,sgn=sort_basis_tuple(tuple(s),self.parent.m_basis_strs)
                                v[E.dwi(s)[2]]+=sgn*adXi[j]
                col_list.append(v)
            self.ad_mat_cache[('E',deg,wght)]=SparseMatrix(list(map(list,col_list))).transpose()
            
        if mod=='CE':
            if ('CE',deg,wght) in self.ad_mat_cache: return None
            C=self.parent.cochain_complex
            if len(C.basis(deg,wght))==0 or len(C.basis(deg,wght+self.wght))==0:
                self.ad_mat_cache[('CE',deg,wght)]=zeros(len(C.basis(deg,wght+self.wght)),
                                                         len(C.basis(deg,wght)))
                return None
            col_list=[]
            for k in range(len(C.basis(deg,wght))):
                W=C.basis(deg,wght)[k] 
                comp=W.components
                # compute ad(self)(W) as a vector
                v=zeros(len(C.basis(deg,wght+self.wght)),1) 

                # Act on the wedge(m^*) part
                for i in range(deg): 
                    adXi=self.ad_mat(mod='m_dual').col(self.parent.m_basis.index(comp[i]))
                    for j in range(len(self.parent.m_basis)):
                        if adXi[j]!=0:
                            s=[str(A) for A in comp[0:-1]]
                            t=s[0:i]+s[i+1:len(s)]
                            if self.parent.m_basis_strs[j] not in t:
                                s[i]=self.parent.m_basis_strs[j]
                                s,sgn=sort_basis_tuple(tuple(s),self.parent.m_basis_strs)
                                s=list(s)+[str(comp[-1])]
                                v[C.dwi(tuple(s))[2]]+=sgn*adXi[j]

                # Act on the g part
                adXi=self.ad_mat().col(self.parent.basis.index(comp[-1]))
                for j in range(len(self.parent.basis)):
                    if adXi[j]!=0:
                        s=[str(A) for A in comp]
                        s[-1]=self.parent.basis_strs[j]
                        v[C.dwi(tuple(s))[2]]+=adXi[j]
                col_list.append(v)
            self.ad_mat_cache[('CE',deg,wght)]=SparseMatrix(list(map(list,col_list))).transpose()

            # Set the scalar cache for wght 0 elts
            if self.wght==0:
                if deg not in self.scalar_cache['CE']: self.scalar_cache['CE'][deg]={}
                A=self.ad_mat('CE',deg,wght)
                self.scalar_cache['CE'][deg][wght] = Matrix([A[i,i] for i in range(shape(A)[0])])
    
    def cast_as_ext_elt(self):
        return self.parent.ext_alg.elt_from_cd({(str(self),):1})
    
    def cast_as_cochain(self):
        return self.parent.cochain_complex.elt_from_cd({(str(self),):1})
    
    def Ad_mat(self,mod='g'):
        """Returns exp(Ad(self)) for the given module; assumes coordinates of the first kind"""
        return exp(self.ad_mat(mod))


# %%
class T_symb(object):
    def __init__(self,basis_strs,wght_list,ad_matrices,Q=None):
        self.basis_strs=basis_strs
        self.wght_list=wght_list
        self.Q=Q
        self.basis=[T_symb_basis_elt(self.basis_strs[i],self.wght_list[i],
                                     self,i,ad_matrices[i]) for i in range(len(self.basis_strs))]
        self.ext_alg=ext_alg(self)
        self.cochain_complex=cochain_complex(self)
        self.m_basis=[self.basis[i] for i in range(len(self.basis)) if self.wght_list[i]<0]
        self.m_basis_strs=[str(A) for A in self.m_basis]
            
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
    
    def mod_dim(self,mod='g',deg=None,wght=None):
        """ Returns the dimension of the specified module, with degree and wght possibly specified
        INPUTS:
        * 'mod' - the desired module, among 'g','m','g_dual','m_dual','E','CE'
        * 'deg' - degree
        * 'wght' - wght
        """
        if deg==None and wght==None:
            if mod in ['g','g_dual']: return len(self.basis)
            if mod in ['m','m_dual']: return len(self.m_basis)
            if deg==None: print('degree must be specified to compute mod_dim of E or CE')
            if mod=='E': return binomial(len(self.m_basis),deg)
            if mod=='CE': return binomial(len(self.m_basis),deg)*len(self.basis)
        
        if deg==None and wght!=None:
            if mod=='g': return len([A for A in self.basis if A.wght==wght])
            if mod=='g_dual': return len([A for A in self.basis if -A.wght==wght])
            if mod=='m': return len([A for A in self.m_basis if A.wght==wght])
            if mod=='m_dual': return len([A for A in self.m_basis if -A.wght==wght])
            if mod in ['E','CE']: raise NotImplementedException
        
        if mod in ['g','g_dual','m','m_dual']: print('do not specify deg when computing dim(m) or dim(g)')
            
        if wght==None:
            if mod=='E': return binomial(len(self.m_basis),deg)
            if mod=='CE': return binomial(len(self.m_basis),deg)*len(self.basis)
        
        if mod=='E': return len(self.ext_alg.basis(deg,wght))
        if mod=='CE': return len(self.cochain_complex.basis(deg,wght))
        
        print('mod_dim module must be among g, m, g_dual, m_dual, E, and CE')
        
    def iprod(self,t1,t2):
        '''t1,t2: of the same type among T_symb_basis_elt, T_symb_elt, ext_elt, and cochain'''
        if not hasattr(t1,'parent') and hasattr(t2,'parent'):
            raise ValueError('arguments of iprod must have type T_symb_basis_elt, T_symb_elt, ext_elt, or cochain')
        
        if self.Q==None: print('Inner product matrix Q not initialized for',self)
        
        if t1==0 or t2==0:
            return 0
        
        if t1.parent==t2.parent:
            return (t1.vec.transpose()*self.Q*t2.vec)[0]
        
        raise invalid_parent_exception('arguments of iprod must have the same parent')
    
    @staticmethod
    def ad(A1,A2):
        return A1.ad(A2)

# %%
class Symp_symb(T_symb):
    def __init__(self,heis_dim):
        if heis_dim<3 or heis_dim%2==0: raise heis_dim_exception
        self.heis_dim=heis_dim
        self.basis_strs=['Y','H','E','X']
        for i in range(1,heis_dim):
            self.basis_strs.append('e_%d'%i)
        self.basis_strs.append('N')
        
        self.wght_list=[1,0,0,-1]
        for i in range(4,len(self.basis_strs)-1):
            self.wght_list.append(-i+3)
        self.wght_list.append(-heis_dim)
        Q=SparseMatrix(diag(*[1,2,2,1]+[sympy.factorial(i-1)/factorial(heis_dim-i-1)
                                        for i in range(1,heis_dim)]+[1]))
        super().__init__(self.basis_strs,self.wght_list,self.ad_mats(),Q)
        
    def ad_mats(self):
        # I'll try to avoid using this
        # Following Medvedev's basis/conventions, except the error in [Y,e_i]=(i-1)*(2*m+1-i)e_{i-1}
        basis_len=len(self.basis_strs)
        Y=SparseMatrix(zeros(basis_len))
        Y[0,1]=2
        Y[1,3]=-1
        for i in range(2,basis_len-3):
            Y[i+2,i+3]=(i-1)*(basis_len-4-i)

        H=SparseMatrix(zeros(basis_len))
        H[0,0]=-2
        H[3,3]=2
        for i in range(1,basis_len-4):
            H[i+3,i+3]=(2*i-basis_len+4)

        X=SparseMatrix(zeros(basis_len))
        X[1,0]=1
        X[3,1]=-2
        for i in range(1,basis_len-5):
            X[i+4,i+3]=1

        E=SparseMatrix(zeros(basis_len))
        for i in range(1,basis_len-4):
            E[i+3,i+3]=1
        E[-1,-1]=2

        ei=[None]+[SparseMatrix(zeros(basis_len)) for i in range(basis_len-5)]

        for i in range(2,len(ei)):
            ei[i][2+i,0]=-(i-1)*(basis_len-4-i)
        for i in range(1,len(ei)):
            ei[i][3+i,1]=-(2*i-basis_len+4)
            ei[i][3+i,2]=-1
            ei[i][-1,basis_len-i-1]=(-1)**i
        for i in range(1,len(ei)-1):
            ei[i][4+i,3]=-1

        N=SparseMatrix(zeros(basis_len))
        N[-1,2]=-2

        return [Y,H,E,X]+ei[1:len(ei)]+[N]
    
       ## Below this point, the methods are specific to the prolonged symbol for the symplectification

    def first_kind_to_second(self,g_elt):
        """Converts an element of G_+ represented in canonical coords of the first kind
        (i.e., exp(eE+hH+yY)) to its representation in canonical coords of the second
        kind (i.e., exp(eE+hH)exp(yY)

        INPUTS:
        * 'v' -- an element of the positive part of self.alg 
        """
        B=self.basis
        v=g_elt.vec
        if v[1]!=0: return (v[0]*(exp(2*v[1])-1)/(2*v[1]))*B[0]+v[1]*B[1]+v[2]*B[2]
        return v[0]*B[0]+v[1]*B[1]+v[2]*B[2]
        
    def second_kind_to_first(self,g_elt):
        """Converts an element of G_+ represented in canonical coords of the 
        second kind (i.e., exp(eE+hH)exp(yY) to its representation in canonical  
        coords of the first kind (i.e., exp(eE+hH+yY))

        INPUTS:
        * 'g_elt' -- an element of the positive part of self.alg 
        """

        if type(g_elt)==T_symb_elt or type(g_elt)==T_symb_basis_elt:
            v=g_elt.vec
        else: v=g_elt
        B=self.basis
        if v[1]!=0: return 2*v[1]*v[0]/(exp(2*v[1])-1)*B[0]+v[1]*B[1]+v[2]*B[2]
        return v[0]*B[0]+v[1]*B[1]+v[2]*B[2]

    def second_kind_inv(self,g_elt):
        """Returns the inverse of and elt of G_+ represented in canonical coordinates
        of the second kind.
        
        INPUTS:
        * 'g_elt' -- an element of the positive part of self.alg
        """
        B=self.basis
        v=g_elt.vec
        return -exp(-2*v[1])*v[0]*B[0]-v[1]*B[1]-v[2]*B[2]

    def second_kind_mul(self,g_elt_1,g_elt_2):
        """Returns g_elt_1*g_elt_2, which is an elt of G_+ represented in canonical coordinates
        of the second kind
        
        INPUTS:
        * 'g_elt_1','g_elt_2' -- elements of the positive part of self.alg
        """
        B=self.basis
        v1=g_elt_1.vec
        v2=g_elt_2.vec
        return (exp(2*v2[1])*v1[0]+v2[0])*B[0]+(v1[1]+v2[1])*B[1]+(v1[2]+v2[2])*B[2]

# %%
def SF_ad(X1,X2,SF):
    '''args: X1,X2 are VFs, either T_elts or vectors/lists with coeffs indexed objects and 
             the coordinates y,h,e SF is the structure function defining the ad-relations between 
             T basis elts, an element of C, the cochain complex
       returns: [X1,X2], where the str function defines the relations between T basis elts,
               and K,y,h,e depend on the coordinates appropriately'''
    T=SF.parent.alg
    result=0
    X=[X1,X2]
    v=[None,None]
    for j in range(2):
        Xj=X[j]
        vj=v[j]
        if type(Xj)==list: v[j]=Xj
        if type(Xj)==type(eye(4)): v[j]=list(Xj)
        if type(Xj)==T_symb_elt or type(Xj)==T_symb_basis_elt: v[j]=Xj.vec_rep
    for i in range(len(v[0])):
        coeff_1=v[0][i]
        if coeff_1!=0:
            for j in range(len(v[1])):
                coeff_2=v[1][j]
                if coeff_2!=0:
                    res1=coeff_1*abn_ind_der(coeff_2,i)*T.basis[j]
                    res2=-coeff_2*abn_ind_der(coeff_1,j)*T.basis[i]
                    new_res=result+res1+res2
                    result+=res1
                    result+=res2
    e_elt=T.elt(v[0]).cast_as_ext_elt().wedge(T.elt(v[1]).cast_as_ext_elt())
    result+=SF.apply_cochain_map(e_elt)
    return result

# %%
class heis_dim_exception(Exception):
    '''Raised when the provided heis_dim is not odd'''
    pass

class invalid_parent_exception(Exception):
    '''Raised when the parent of an argument isn't what it should be'''  
    pass

class constr_Mat_size_exception(Exception):
    """Raised when a constructor receives a Matrix of incorrect size"""
    pass

class arg_required_exception(Exception):
    """Raised when a required argument is omitted"""
    pass