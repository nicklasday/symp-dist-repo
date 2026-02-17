all=["VectorInput"]

from sympy import Matrix, Expr
from typing import Union, Sequence,TYPE_CHECKING
if TYPE_CHECKING:
    from ..distributions import Vector_Field

VectorInput = Union[Sequence[Expr], Matrix, 'Vector_Field']