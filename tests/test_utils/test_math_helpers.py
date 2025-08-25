import unittest
from src.utils import math_helpers as mh
from src.utils import coeff_dict_helpers as cdh

class helpers_test(unittest.TestCase):
    def test_str_from_coeff_dict(self):
        # This also tests remove_zeros
        cd:list[dict]
        cd.append({})
        cd.append({'A':0})
        cd.append({'A':1})
        cd.append({'A':-1})
        cd.append({'A':2})
        cd.append({'A':1,'B':1})
        cd.append({'A':1,'B':-1,'C':0})
        cd.append({'A':-1,'B':2,'D':0})
        cd.append({'A':2,'B':1})
        cd.append({('A','B'):3,('C',):-4})
        cd.append({'A':0,('B','C'):0})
        cd.append({('A','B','C'):0,'D':-1})
        cd.append({('A','B','C'):1,('D','E'):-1})
        cd.append({('A','B','C'):-1,('D','E'):1})
        
        self.assertEqual(cdh.str_from_coeff_dict(cd[0]),'0')
        self.assertEqual(cdh.str_from_coeff_dict(cd[1]),'0')
        self.assertEqual(cdh.str_from_coeff_dict(cd[2]),'A')
        self.assertEqual(cdh.str_from_coeff_dict(cd[3]),'- A')
        self.assertEqual(cdh.str_from_coeff_dict(cd[4]),'2*A')
        self.assertEqual(cdh.str_from_coeff_dict(cd[5]),'A + B')
        self.assertEqual(cdh.str_from_coeff_dict(cd[6]),'A - B')
        self.assertEqual(cdh.str_from_coeff_dict(cd[7]),'- A + 2*B')
        self.assertEqual(cdh.str_from_coeff_dict(cd[8]),'2*A + B')
        self.assertEqual(cdh.str_from_coeff_dict(cd[9]),'3*(A,B) + -4*(C)')
        self.assertEqual(cdh.str_from_coeff_dict(cd[10]),'0')
        self.assertEqual(cdh.str_from_coeff_dict(cd[11]),'- D')
        self.assertEqual(cdh.str_from_coeff_dict(cd[12]),'(A,B,C) - (D,E)')
        self.assertEqual(cdh.str_from_coeff_dict(cd[13]),'- (A,B,C) + (D,E)')
        
#     def test_find_cochain_basis(self):
#         T5=SympSymb(5)
#         C5=T5.cochain_complex
#         c0=C5.elt({})
#         c1=C5.elt_from_cd({('e_1','X','e_1'):1,('e_2','X','Y'):-1})
#         c2=-c1
#         c3=C5.elt_from_cd({('N','X','e_1'):1,('e_4','X','e_2'):2})
#         c4=C5.elt_from_cd({('e_2','X','e_1'):1,('e_3','X','e_2'):1})
#         c5=C5.elt({2:{2:SparseMatrix([[1]+[0]*15]).transpose}})
        
#         t1=find_cochain_basis([c1,c2,c3,c4])
#         t1.sort()
#         r1=[C5.cochain({('Y','X','Y'):1}),C5.cochain({('Y','X','e_2'):1}),C5.cochain({('Y','X','e_1'):1})]
#         r1.sort()
#         self.assertEqual(t1,r1)
        
    def test_sort_basis_tuple(self):
        b0:tuple=tuple()
        b1:tuple=('X','Y')
        b2:tuple=('Y','X')
        b3:tuple=('X','Y','e_1','e_3','H')
        b4:tuple=('X','Y','e_1','e_3','H','banana')
        
        b=['Y','H','E','X','e_1','e_2','e_3','N']
        
        self.assertEqual(mh.sort_basis_tuple(b0,b),(tuple(),1))
        self.assertEqual(mh.sort_basis_tuple(b1,b),(('Y','X'),-1))
        self.assertEqual(mh.sort_basis_tuple(b2,b),(('Y','X'),1))
        self.assertEqual(mh.sort_basis_tuple(b3,b),(('Y','H','X','e_1','e_3'),1))
        self.assertRaises(ValueError,mh.sort_basis_tuple,b4,b)