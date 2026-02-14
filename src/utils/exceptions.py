__all__ = [
    "heis_dim_exception",
    "invalid_parent_exception",
    "constr_Mat_size_exception",
    "arg_required_exception",
    "invalid_ext_elt_casting",
    "cochain_init_Exception",
    "nonhomogeneous_Exception",
    "no_linear_monomial_exception",
]


class heis_dim_exception(Exception):
    """Raised when the provided heis_dim is not odd"""

    pass


class invalid_parent_exception(Exception):
    """Raised when the parent of an argument isn't what it should be"""

    pass


class constr_Mat_size_exception(Exception):
    """Raised when a constructor receives a Matrix of incorrect size"""

    pass


class arg_required_exception(Exception):
    """Raised when a required argument is omitted"""

    pass


class invalid_ext_elt_casting(Exception):
    """T_symb_elt with positive component cannot be cast as ext_elt"""

    pass


class cochain_init_Exception(Exception):
    def __init__(self, message=""):
        self.message = message
        super().__init__(self.message)


class nonhomogeneous_Exception(Exception):
    def __init__(self, message=""):
        self.message = message
        super().__init__(self.message)


class no_linear_monomial_exception(Exception):
    def __init__(self, message=""):
        self.message = message
        super().__init__(self.message)


class wedge_Mat_Exception(Exception):
    def __init__(self, A, message="wedge_Mat failure:"):
        self.message = message + str(A)
        super().__init__(self.message)

class Coordinatization_Exception(Exception):
    def __init__(self,message="Failure in coordinatization"):
        self.message = message
        super().__init__(self.message)

class Failed_Check_Exception(Exception):
    def __init__(self,message="Failed Check"):
        self.message = message
        super().__init__(self.message)