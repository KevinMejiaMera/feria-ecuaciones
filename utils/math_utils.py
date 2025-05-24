from sympy import latex, Derivative, diff, integrate
from sympy.core.function import AppliedUndef
from sympy.core.basic import Basic
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
transformations = (standard_transformations + (implicit_multiplication_application,))
def safe_latex(expr, default="\text{Expresión no válida}"):
    """
    Convierte una expresión a formato LaTeX con manejo de errores robusto.
    
    Args:
        expr: Expresión SymPy o string a convertir
        default: Texto alternativo si la conversión falla
        
    Returns:
        String en formato LaTeX
    """
    try:
        if isinstance(expr, str):
            # Intenta parsear la expresión primero
            parsed_expr = parse_expr(expr, transformations=transformations)
            return latex(parsed_expr)
        else:
            return latex(expr)
    except Exception as e:
        print(f"Error en safe_latex: {str(e)}")
        return default
def classify_equation(eq_diff, y, x):
    """Classify the type of differential equation"""
    try:
        if eq_diff.is_linear():
            return "Lineal"
        elif eq_diff.is_separable:
            return "Separable"
        elif eq_diff.is_exact:
            return "Exacta"
        elif eq_diff.is_homogeneous:
            return "Homogénea"
        else:
            return "No lineal"
    except:
        return "No identificada"

def explain_solution_steps(eq_diff, y, x, sol, is_general=True):
    """Generate explanation steps for the solution"""
    pasos = []
    tipo = classify_equation(eq_diff, y, x)
    pasos.append(f"La ecuación es de tipo: {tipo}")
    
    if tipo == "Lineal":
        pass  # Add specific steps for linear equations
    
    pasos.append(f"Solución {'general' if is_general else 'particular'}: {safe_latex(sol)}")
    return pasos