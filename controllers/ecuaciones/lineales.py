from flask import jsonify
from sympy import symbols, Function, Eq, dsolve, Derivative, classify_ode, latex, exp, sin, cos, simplify
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from utils.math_utils import safe_latex
from utils.plot_utils import generate_plot

transformations = (standard_transformations + (implicit_multiplication_application,))

def preprocess_equation(eq_str):
    """Preprocesa la ecuación para facilitar el parsing"""
    replacements = {
        '^': '**',
        'e^': 'exp(',
        'e**': 'exp(',
        'y\'': 'Derivative(y, x)',
        'y\'\'': 'Derivative(y, x, x)',
        'y\'\'\'': 'Derivative(y, x, x, x)',
        'dy/dx': 'Derivative(y, x)',
        'd²y/dx²': 'Derivative(y, x, x)',
        'd³y/dx³': 'Derivative(y, x, x, x)'
    }
    
    for old, new in replacements.items():
        eq_str = eq_str.replace(old, new)
    
    return eq_str.replace('exp(x)', 'exp(x)').replace('exp**x', 'exp(x)')

def validate_equation(eq, y):
    """Valida que la ecuación sea resoluble"""
    if not eq.has(Derivative):
        raise ValueError("La ecuación debe contener derivadas de y (use y' o y'')")
    
    unsupported = ['z', 't', 'w']
    for sym in unsupported:
        if eq.has(sym):
            raise ValueError(f"Símbolo no soportado: '{sym}'. Use solo 'x' e 'y'")

def parse_conditions(conditions_str, x, y):
    """Parsea las condiciones iniciales"""
    ics = {}
    applied_conditions = []
    
    if not conditions_str:
        return ics, applied_conditions
    
    for cond in [c.strip() for c in conditions_str.split(',') if c.strip()]:
        if '=' not in cond:
            continue
            
        lhs_cond, rhs_cond = cond.split('=', 1)
        lhs_cond = lhs_cond.strip()
        rhs_cond = rhs_cond.strip()

        try:
            rhs_val = float(rhs_cond) if '.' in rhs_cond else int(rhs_cond)
        except ValueError:
            try:
                rhs_val = parse_expr(rhs_cond.replace('^', '**'), {'x': x}, transformations=transformations)
            except:
                rhs_val = rhs_cond

        if '(' in lhs_cond and ')' in lhs_cond:
            func_part = lhs_cond.split('(')[0].strip()
            x_val_str = lhs_cond.split('(')[1].split(')')[0]
            
            try:
                x_val = parse_expr(x_val_str, {'x': x}, transformations=transformations)
                
                if "'" in func_part:
                    deriv_order = func_part.count("'")
                    ics[Derivative(y, x, deriv_order).subs(x, x_val)] = rhs_val
                    cond_type = f"Derivada de orden {deriv_order}"
                else:
                    ics[y.subs(x, x_val)] = rhs_val
                    cond_type = "Valor inicial"
                
                applied_conditions.append({
                    'condition': cond,
                    'type': cond_type,
                    'x_value': float(x_val.evalf()) if hasattr(x_val, 'evalf') else str(x_val),
                    'y_value': float(rhs_val.evalf()) if hasattr(rhs_val, 'evalf') else str(rhs_val)
                })
            except Exception as e:
                raise ValueError(f"Error al parsear condición '{cond}': {str(e)}")
    
    return ics, applied_conditions

def get_solution_method(classifications):
    """Determina el método de solución basado en la clasificación"""
    if '1st_linear' in classifications:
        return {
            'name': 'Factor Integrante',
            'steps': [
                "Identificar P(x) y Q(x) en y' + P(x)y = Q(x)",
                "Calcular el factor integrante μ(x) = e^(∫P(x)dx)",
                "Multiplicar toda la ecuación por μ(x)",
                "Simplificar el lado izquierdo como d/dx[μ(x)y]",
                "Integrar ambos lados con respecto a x",
                "Despejar y(x) para obtener la solución general"
            ]
        }
    elif 'Bernoulli' in classifications:
        return {
            'name': 'Bernoulli',
            'steps': [
                "Reescribir la ecuación en la forma y' + P(x)y = Q(x)y^n",
                "Hacer la sustitución v = y^(1-n)",
                "Resolver la ecuación lineal resultante para v",
                "Sustituir de vuelta v = y^(1-n) para obtener y(x)"
            ]
        }
    elif '2nd_linear' in classifications:
        return {
            'name': 'Coeficientes Constantes',
            'steps': [
                "Encontrar la ecuación característica",
                "Calcular las raíces de la ecuación característica",
                "Determinar la solución homogénea basada en las raíces",
                "Encontrar una solución particular usando coeficientes indeterminados",
                "Sumar soluciones homogénea y particular"
            ]
        }
    else:
        return {
            'name': 'Método General',
            'steps': ["Se aplicó un método general de resolución"]
        }

def resolver_lineales(data):
    try:
        # Validación inicial
        equation_str = data.get('equation', '').strip()
        conditions_str = data.get('conditions', '').strip()
        
        if not equation_str:
            return jsonify({
                'status': 'error',
                'error': 'empty_equation',
                'message': 'La ecuación no puede estar vacía',
                'suggestions': ['Ingrese una ecuación como: y\' + y = 0']
            }), 400

        # Configurar símbolos
        x = symbols('x')
        y = Function('y')(x)
        pasos = []
        response_data = {
            'status': 'success',
            'order': None,
            'type': 'linear',
            'steps': []
        }

        # Parsear y validar la ecuación
        try:
            processed_eq = preprocess_equation(equation_str)
            
            if '=' in processed_eq:
                lhs, rhs = processed_eq.split('=', 1)
                lhs_expr = parse_expr(lhs, {'y': y, 'x': x}, transformations=transformations)
                rhs_expr = parse_expr(rhs, {'y': y, 'x': x}, transformations=transformations)
                eq = Eq(lhs_expr, rhs_expr)
            else:
                expr = parse_expr(processed_eq.strip(), {'y': y, 'x': x}, transformations=transformations)
                eq = Eq(expr, 0)
                
            validate_equation(eq, y)
            
        except Exception as e:
            error_msg = str(e)
            if "Missing \left or extra \right" in error_msg:
                error_msg = "La ecuación contiene paréntesis o llaves no balanceados"
            elif "Derivative" in error_msg:
                error_msg = "Error en la especificación de derivadas. Use y' para primera derivada, y'' para segunda derivada, etc."
            
            return jsonify({
                'status': 'error',
                'error': 'parse_error',
                'message': 'Error en el formato de la ecuación',
                'details': error_msg,
                'suggestions': [
                    "Formato correcto para primer orden: y' + 2y = e^x",
                    "Para segundo orden: y'' - 3y' + 2y = sin(x)",
                    "Use y(0)=1 para condiciones iniciales"
                ]
            }), 400

        # Determinar tipo y orden
        try:
            order = len(eq.lhs.find(Derivative))
            classifications = classify_ode(eq, y)
            
            response_data['order'] = order
            response_data['classification'] = classifications
            
            if not any(t in classifications for t in ['1st_linear', '2nd_linear', 'nth_linear']):
                return jsonify({
                    'status': 'error',
                    'error': 'non_linear_equation',
                    'message': 'La ecuación no es lineal o no es soportada',
                    'suggestions': [
                        "Formato para primer orden: y' + P(x)y = Q(x)",
                        "Formato para segundo orden: y'' + P(x)y' + Q(x)y = R(x)",
                        "Ejemplo válido: y'' - 3y' + 2y = sin(x)"
                    ]
                }), 400
                
            method_info = get_solution_method(classifications)
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error': 'classification_error',
                'message': 'Error al clasificar la ecuación',
                'details': str(e)
            }), 400

        # Resolver ecuación general
        try:
            sol_general = dsolve(eq, y)
            simplified_sol = simplify(sol_general)
            
            response_data['solution'] = safe_latex(sol_general)
            response_data['simplified_solution'] = safe_latex(simplified_sol)
            response_data['needs_conditions'] = 'C' in str(sol_general)
            
            method_steps = method_info['steps']
            solution_steps = []
            
            for i, step_desc in enumerate(method_steps, start=1):
                solution_steps.append({
                    'step': i,
                    'description': f"Paso {i} del método {method_info['name']}",
                    'details': step_desc,
                    'method': method_info['name']
                })
            
            solution_steps.append({
                'step': len(method_steps) + 1,
                'description': 'Solución general encontrada',
                'method': method_info['name'],
                'solution': safe_latex(sol_general),
                'simplified_solution': safe_latex(simplified_sol),
                'details': f"Se aplicó el método de {method_info['name']} para obtener la solución general"
            })
            
            pasos.extend(solution_steps)
            
        except Exception as e:
            error_msg = f"No se pudo resolver la ecuación. Error: {str(e)}"
            return jsonify({
                'status': 'error',
                'error': 'solve_error',
                'message': 'Error al resolver la ecuación',
                'details': error_msg,
                'suggestions': [
                    "Verifique que la ecuación esté correctamente escrita",
                    "Asegúrese de usar solo 'x' como variable independiente",
                    "Ejemplo válido: y' + 2y = e^x"
                ]
            }), 400

        # Procesar condiciones iniciales (sección mejorada)
        if conditions_str:
            try:
                ics, applied_conditions = parse_conditions(conditions_str, x, y)
                additional_steps = []
                
                if ics:
                    sol_particular = dsolve(eq, y, ics=ics)
                    simplified_particular = simplify(sol_particular)
                    
                    response_data['particular_solution'] = safe_latex(sol_particular)
                    response_data['simplified_particular'] = safe_latex(simplified_particular)
                    response_data['conditions'] = applied_conditions
                    
                    # Generación de gráfico mejorada
                    plot_html = generate_plot(eq, sol_particular, [ics])
                    if plot_html:
                        response_data['plot_html'] = plot_html
                    else:
                        response_data['plot_info'] = {
                            'message': 'El gráfico no está disponible para esta solución',
                            'reason': 'La solución puede contener singularidades o funciones no soportadas',
                            'solution': safe_latex(sol_particular)
                        }
                    
                    # Pasos adicionales para condiciones iniciales
                    start_step = len(pasos) + 1
                    
                    additional_steps.append({
                        'step': start_step,
                        'description': 'Aplicación de condiciones iniciales',
                        'details': f"Se aplicaron las siguientes condiciones:\n" + 
                                   "\n".join([f"- {c['condition']} ({c['type']})" for c in applied_conditions]),
                        'conditions': applied_conditions,
                        'method': method_info['name']
                    })
                    
                    for i, cond in enumerate(applied_conditions, start=1):
                        additional_steps.append({
                            'step': start_step + i,
                            'description': f'Sustitución de condición {cond["condition"]}',
                            'details': f"Se sustituyó x = {cond['x_value']}, y = {cond['y_value']} en la solución general",
                            'method': method_info['name']
                        })
                    
                    additional_steps.append({
                        'step': start_step + len(applied_conditions) + 1,
                        'description': 'Solución particular encontrada',
                        'solution': safe_latex(sol_particular),
                        'simplified_solution': safe_latex(simplified_particular),
                        'details': "Se resolvió el sistema para las constantes arbitrarias",
                        'method': method_info['name']
                    })
                    
                    response_data['additional_steps'] = additional_steps
                    
            except Exception as e:
                pasos.append({
                    'step': len(pasos) + 1,
                    'description': 'Error aplicando condiciones iniciales',
                    'error': str(e),
                    'suggestions': [
                        "Formato correcto: y(0)=1, y'(0)=2",
                        "Separe múltiples condiciones con comas",
                        "Ejemplo válido: y(0)=1, y'(0)=0"
                    ],
                    'method': method_info['name']
                })
                response_data['warning'] = f"Error en condiciones: {str(e)}"

        # Preparar respuesta final
        response_data['steps'] = pasos
        
        return jsonify(response_data)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': 'internal_error',
            'message': 'Error interno en el servidor',
            'details': str(e)
        }), 500