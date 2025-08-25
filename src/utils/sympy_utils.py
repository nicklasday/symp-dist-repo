from __future__ import annotations
from typing import TYPE_CHECKING
from sympy import Matrix, MatrixBase, SparseMatrix

if TYPE_CHECKING:
    from .types import VectorInput

def to_sympy_matrix(v: VectorInput) -> Matrix:
    from ..distributions import Vector_Field
    if isinstance(v,Vector_Field):
        return Matrix(v.vec)
    return Matrix(v) if not isinstance(v, MatrixBase) else v

def to_sympy_sparse_matrix(v: VectorInput) -> Matrix:
    from ..distributions import Vector_Field
    if isinstance(v,Vector_Field):
        return SparseMatrix(v.vec)
    return SparseMatrix(v) if not isinstance(v,SparseMatrix) else v