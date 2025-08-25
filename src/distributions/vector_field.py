__all__ = ["Vector_Field"]
import sympy as sp


class Vector_Field(object):
    def __init__(self, v_rep, parent):
        """INPUTS:
        * 'v_rep' - a list or vector representing a distribution in the frame of parent (positive and negative)
        * 'parent' - a Distr_of_constant_symbol object"""
        self.vec = sp.Matrix(v_rep)
        if sp.shape(self.vec)[1] != 1:
            self.vec = self.vec.transpose()
        self.parent = parent

    def bracket(self, other):
        return self.parent.bracket(self, other)

    def __str__(self):
        return str(self.vec)

    def __repr__(self):
        return self.vec.__repr__()

    def __add__(self, other):
        return Vector_Field(self.vec + other.vec, self.parent)

    def __neg__(self):
        return Vector_Field(-self.vec, self.parent)

    def __sub__(self, other):
        return self + (-other)
