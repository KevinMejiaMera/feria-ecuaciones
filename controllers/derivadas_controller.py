from flask import jsonify
from sympy import symbols, diff, latex
from sympy.parsing.sympy_parser import parse_expr
from utils.equation_parser import transformations
import sympy as sp
from sympy import symbols, Function, Eq, dsolve, diff, Derivative, exp, integrate, sin, cos, tan, log, lambdify
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,                                  implicit_multiplication_application)
import numpy as np
import plotly.graph_objects as go
import re
from sympy import classify_ode 
from sympy import latex
from utils.equation_parser import parse_equation
from utils.equation_parser import safe_parse
from utils.math_utils import safe_latex
from utils.plot_utils import generate_plot

def calcular_derivada(data):
    try:
        funcion = data['funcion']
        variable = data.get('variable', 'x')
        
        funcion = funcion.replace('^', '**')
        x = symbols(variable)
        expr = parse_expr(funcion, {variable: x}, transformations=transformations)
        derivada = diff(expr, x)
        
        pasos = generar_pasos_derivacion(expr, derivada, x)
        
        return jsonify({
            'derivada': latex(derivada),
            'pasos': pasos,
            'expr_original': latex(expr)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def generar_pasos_derivacion(expr, derivada, x):
    pasos = []
    original_expr = latex(expr)
    pasos.append(f"Expresión original: ${original_expr}$")
    
    if expr.is_Add:
        pasos.append("Aplicando regla de la suma (derivada de una suma es la suma de las derivadas):")
        for term in expr.args:
            term_deriv = diff(term, x)
            pasos.append(f"Derivada de ${latex(term)}$ es ${latex(term_deriv)}$")
    
    elif expr.is_Pow:
        base, exponent = expr.as_base_exp()
        if base == x:
            pasos.append(f"Aplicando regla de la potencia ($\\frac{{d}}{{dx}}x^n = nx^{{n-1}}$):")
            pasos.append(f"$\\frac{{d}}{{dx}}{latex(expr)} = {latex(exponent)}x^{{{latex(exponent-1)}}}$")
    
    elif expr.is_Function:
        if isinstance(expr, sin):
            pasos.append("Aplicando derivada de seno ($\\frac{d}{dx}\\sin(x) = \\cos(x)$)")
        elif isinstance(expr, cos):
            pasos.append("Aplicando derivada de coseno ($\\frac{d}{dx}\\cos(x) = -\\sin(x)$)")
        elif isinstance(expr, exp):
            pasos.append("Aplicando derivada de exponencial ($\\frac{d}{dx}e^x = e^x$)")
    
    pasos.append(f"Resultado final: ${latex(derivada)}$")
    return pasos
# Añade esto al final del archivo:
if __name__ == '__main__':
    # Solo para pruebas
    test_data = {'funcion': 'x^2', 'variable': 'x'}
    print(calcular_derivada(test_data).get_json())