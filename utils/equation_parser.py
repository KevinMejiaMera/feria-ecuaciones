from sympy import symbols, Function, Eq, Derivative, parse_expr
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application

transformations = (standard_transformations + (implicit_multiplication_application,))

def parse_equation(eq_str, y_str="y", x_str="x"):
    """Parse a differential equation string into a SymPy equation"""
    eq_str = eq_str.replace('^', '**').replace('exp', 'exp')
    eq_str = eq_str.replace("y'", f"Derivative({y_str}({x_str}), {x_str})")
    eq_str = eq_str.replace("y''", f"Derivative({y_str}({x_str}), ({x_str}, 2))")
    return eq_str

def safe_parse(expr_str, variables):
    """Safely parse an expression with error handling"""
    try:
        return parse_expr(expr_str, variables, transformations=transformations)
    except Exception as e:
        raise ValueError(f"Error al parsear la expresión: {str(e)}")