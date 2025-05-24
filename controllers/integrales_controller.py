from flask import jsonify
from sympy import symbols, integrate, latex, sin, cos, exp, log
from sympy.parsing.sympy_parser import parse_expr
from utils.equation_parser import transformations
from utils.equation_parser import parse_equation
from utils.equation_parser import safe_parse
from utils.math_utils import safe_latex
from utils.plot_utils import generate_plot

def calcular_integral(data):
    try:
        funcion = data['funcion']
        variable = data.get('variable', 'x')
        
        funcion = funcion.replace('^', '**')
        x = symbols(variable)
        expr = parse_expr(funcion, {variable: x}, transformations=transformations)
        integral = integrate(expr, x)
        
        pasos = generar_pasos_integracion(expr, integral, x)
        
        return jsonify({
            'integral': latex(integral),
            'pasos': pasos,
            'expr_original': latex(expr)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def generar_pasos_integracion(expr, integral, x):
    pasos = []
    original_expr = latex(expr)
    pasos.append(f"Expresión original: \\[{original_expr}\\]")
    
    if expr.is_Add:
        pasos.append("Aplicando regla de la suma (integral de una suma es la suma de las integrales):")
        for term in expr.args:
            term_int = integrate(term, x)
            pasos.append(f"Integral de ${latex(term)}$ es ${latex(term_int)}$")
    
    elif expr.is_Pow:
        base, exponent = expr.as_base_exp()
        if base == x:
            pasos.append("Aplicando regla de la potencia para integrales:")
            pasos.append(f"\\[\\int x^{{{latex(exponent)}}} \\, dx = \\frac{{x^{{{latex(exponent+1)}}}}}{{{latex(exponent+1)}}} + C \\]")
    
    elif expr.is_Function:
        if isinstance(expr, sin):
            pasos.append("Aplicando integral de seno:")
            pasos.append("\\[\\int \\sin(x) \\, dx = -\\cos(x) + C\\]")
        elif isinstance(expr, cos):
            pasos.append("Aplicando integral de coseno:")
            pasos.append("\\[\\int \\cos(x) \\, dx = \\sin(x) + C\\]")
        elif isinstance(expr, exp):
            pasos.append("Aplicando integral de exponencial:")
            pasos.append("\\[\\int e^x \\, dx = e^x + C\\]")
        elif isinstance(expr, log):
            pasos.append("Aplicando integral de logaritmo natural:")
            pasos.append("\\[\\int \\frac{{1}}{{x}} \\, dx = \\ln|x| + C\\]")
    
    elif expr.is_Mul:
        pasos.append("Considerar método de integración por partes:")
        pasos.append("\\[\\int u \\, dv = uv - \\int v \\, du\\]")
    
    pasos.append(f"Resultado final: \\[{latex(integral)} + C\\]")
    return pasos