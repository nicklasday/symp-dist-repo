import unittest
import sympy as sp
from symp_dist.algebra.tanaka_symbols.symplectic_symbol import SympSymb
from symp_dist.distributions import Free_distr
from symp_dist.cartan_geometries import Geom_Prolongation

class Distribution_of_constant_symbol_test(unittest.TestCase):
    def setUp(self):
        self.K=sp.IndexedBase('K')
        self.T=SympSymb(7)
        self.C=self.T.cochain_complex
        # self.D=Free_distr(self.T,7)[0]
        self.Free_D=Free_distr(self.T)[0]
        self.P=Geom_Prolongation(self.Free_D,{})
    
    # def test_normalize_der_Jacobi(self):
    #     test_args=[self.K[0,0,0,5]]
    #     test_vals=[self.K[0,0,0,4,3]-self.K[0,0,0,3,4]-self.K[3,4,3]*self.K[0,0,0,3]-self.K[3,4,4]*self.K[0,0,0,4]]

    #     test_args.append(self.K[0,0,0,3])
    #     test_vals.append(self.K[0,0,0,3])

    #     test_args.append(self.K[0,0,0,4])
    #     test_vals.append(self.K[0,0,0,4])

    #     test_args.append(self.K[0,0,0,5,3])
    #     test_vals.append(self.K[0,0,0,4,3,3]-self.K[0,0,0,3,4,3]-self.K[3,4,3,3]*self.K[0,0,0,3]-self.K[3,4,3]*self.K[0,0,0,3,3]-self.K[3,4,4,3]*self.K[0,0,0,4]-self.K[3,4,4]*self.K[0,0,0,4,3])

    #     test_args.append(self.K[0,0,0,3,5])
    #     test_vals.append(self.K[0,0,0,3,4,3]-self.K[0,0,0,3,3,4]-self.K[3,4,3]*self.K[0,0,0,3,3]-self.K[3,4,4]*self.K[0,0,0,3,4])

    #     test_args.append(self.K[0,0,0,6])
    #     test_vals.append((self.D.normalize_der(self.K[0,0,0,5,3])-self.D.normalize_der(self.K[0,0,0,3,5])
    #                     -self.K[3,5,3]*self.K[0,0,0,3]-self.K[3,5,4]*self.K[0,0,0,4]-self.K[3,5,5]*self.D.normalize_der(self.K[0,0,0,5])).subs(JS_dict))

    #     test_args.append(self.K[0,0,0,7])
    #     test_vals.append((self.D.normalize_der(K[0,0,0,6,3])-self.D.normalize_der(K[0,0,0,3,6])-K[3,6,3]*K[0,0,0,3]-K[3,6,4]*K[0,0,0,4]-K[3,6,5]*self.D.normalize_der(K[0,0,0,5])
    #              -K[3,6,6]*self.D.normalize_der(K[0,0,0,6])).subs(JS_dict))

    #     test_args.append(self.K[0,0,0,5]**3)
    #     test_vals.append(self.D.normalize_der(self.K[0,0,0,5])**3)

    #     test_args.append(self.K[0,0,0,5]*self.K[0,0,0,3,5])
    #     test_vals.append(self.D.normalize_der(self.K[0,0,0,5])*self.D.normalize_der(self.K[0,0,0,3,5]))

    #     test_args.append(3*self.K[0,0,0,3,5])
    #     test_vals.append(3*self.D.normalize_der(self.K[0,0,0,3,5]))

    #     test_args.append(self.K[0,0,0,3,5]+self.K[0,0,0,6])
    #     test_vals.append(self.D.normalize_der(self.K[0,0,0,3,5])+self.D.normalize_der(self.K[0,0,0,6]))

    #     test_args.append(self.K[0,0,0,3,5]**self.K[0,0,0,5])
    #     test_vals.append(self.D.normalize_der(self.K[0,0,0,3,5])**self.D.normalize_der(self.K[0,0,0,5]))

    #     test_args.append(symbols('x')*self.K[0,0,0,3,7])
    #     test_vals.append(symbols('x')*self.D.normalize_der(self.K[0,0,0,3,7]))

    #     test_args.append(self.K[0,0,0,5,5])
    #     test_vals.append(self.D.normalize_der(self.D.abn_ind_der(self.D.normalize_der(self.K[0,0,0,5]),5)))

    #     for i in range(len(test_args)):
    #         self.assertEqual(simplify(self.D.normalize_der(test_args[i])-test_vals[i]),0)

    def test_normalize_der_Free(self):
        test_args=[self.K[0,0,0,5]]
        test_vals=[self.K[0,0,0,4,3]-self.K[0,0,0,3,4]-self.K[3,4,3]*self.K[0,0,0,3]-self.K[3,4,4]*self.K[0,0,0,4]]

        test_args.append(self.K[0,0,0,3])
        test_vals.append(self.K[0,0,0,3])

        test_args.append(self.K[0,0,0,4])
        test_vals.append(self.K[0,0,0,4])

        test_args.append(self.K[0,0,0,5,3])
        test_vals.append(self.K[0,0,0,4,3,3]-self.K[0,0,0,3,4,3]-self.K[3,4,3,3]*self.K[0,0,0,3]-self.K[3,4,3]*self.K[0,0,0,3,3]-self.K[3,4,4,3]*self.K[0,0,0,4]-self.K[3,4,4]*self.K[0,0,0,4,3])

        test_args.append(self.K[0,0,0,3,5])
        test_vals.append(self.K[0,0,0,3,4,3]-self.K[0,0,0,3,3,4]-self.K[3,4,3]*self.K[0,0,0,3,3]-self.K[3,4,4]*self.K[0,0,0,3,4])

        test_args.append(self.K[0,0,0,6])
        test_vals.append((self.Free_D.normalize_der(self.K[0,0,0,5,3])-self.Free_D.normalize_der(self.K[0,0,0,3,5])
                        -self.K[3,5,3]*self.K[0,0,0,3]-self.K[3,5,4]*self.K[0,0,0,4]-self.K[3,5,5]*self.Free_D.normalize_der(self.K[0,0,0,5])))

        test_args.append(self.K[0,0,0,7])
        test_vals.append((self.Free_D.normalize_der(self.K[0,0,0,6,3])-self.Free_D.normalize_der(self.K[0,0,0,3,6])-self.K[3,6,3]*self.K[0,0,0,3]-self.K[3,6,4]*self.K[0,0,0,4]-self.K[3,6,5]*self.Free_D.normalize_der(self.K[0,0,0,5])
                        -self.K[3,6,6]*self.Free_D.normalize_der(self.K[0,0,0,6])))

        test_args.append(self.K[0,0,0,5]**3)
        test_vals.append(self.Free_D.normalize_der(self.K[0,0,0,5])**3)

        test_args.append(self.K[0,0,0,5]*self.K[0,0,0,3,5])
        test_vals.append(self.Free_D.normalize_der(self.K[0,0,0,5])*self.Free_D.normalize_der(self.K[0,0,0,3,5]))

        test_args.append(3*self.K[0,0,0,3,5])
        test_vals.append(3*self.Free_D.normalize_der(self.K[0,0,0,3,5]))

        test_args.append(self.K[0,0,0,3,5]+self.K[0,0,0,6])
        test_vals.append(self.Free_D.normalize_der(self.K[0,0,0,3,5])+self.Free_D.normalize_der(self.K[0,0,0,6]))

        test_args.append(self.K[0,0,0,3,5]**self.K[0,0,0,5])
        test_vals.append(self.Free_D.normalize_der(self.K[0,0,0,3,5])**self.Free_D.normalize_der(self.K[0,0,0,5]))

        test_args.append(sp.symbols('x')*self.K[0,0,0,3,7])
        test_vals.append(sp.symbols('x')*self.Free_D.normalize_der(self.K[0,0,0,3,7]))

        test_args.append(self.K[0,0,0,5,5])
        test_vals.append(self.Free_D.normalize_der(self.Free_D.abn_ind_der(self.Free_D.normalize_der(self.K[0,0,0,5]),5)))

        for i in range(len(test_args)):
            self.assertEqual(sp.simplify(self.Free_D.normalize_der(test_args[i])-test_vals[i]),0)
            
    # We pass this, but it takes 35 min...
    # I should probably figure out why it takes that long
    # def test_Jacobi_subs(self):
    #     for A in JS_dict:
    #         if Indexed_obj_in_expr(JS_dict[A]).intersection(set(JS_dict.keys()))!=set():
    #             print('Substitutions needed')
    #             self.assertTrue(False)
        
    #     wght_1_list=[]
    #     wght_2_list=[]
    #     wght_3_list=[]
    #     for i in range(3,len(self.T.basis)):
    #         for j in range(i+1,len(self.T.basis)):
    #             for k in range(j+1,len(self.T.basis)):
    #                 for l in range(3,len(self.T.basis)):
    #                     if self.T.basis[i].wght+self.T.basis[j].wght+self.T.basis[k].wght-self.T.basis[l].wght==-1:
    #                         wght_1_list.append((i,j,k,l))
    #                     if self.T.basis[i].wght+self.T.basis[j].wght+self.T.basis[k].wght-self.T.basis[l].wght==-2:
    #                         wght_2_list.append((i,j,k,l))
    #                     if self.T.basis[i].wght+self.T.basis[j].wght+self.T.basis[k].wght-self.T.basis[l].wght==-3:
    #                         wght_3_list.append((i,j,k,l))

    #     for A in wght_1_list+wght_2_list+wght_3_list:
    #         print(A)
    #         self.assertEqual(simplify(self.D.Jacobi_id(*A).subs(JS_dict)),0)
