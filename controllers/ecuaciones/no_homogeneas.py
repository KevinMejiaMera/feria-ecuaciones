import re
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication, convert_xor
import traceback
from utils.math_utils import safe_latex
from sympy import Eq, lambdify
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from io import BytesIO
import base64


class ResolvedorEcuacionesNoHomogeneas:
    def __init__(self):  # Corregido: __init__ en lugar de _init_
        self.x = sp.Symbol('x')
        self.y = sp.Function('y')(self.x)  # Corregido: y como función de x
        self.C1 = sp.Symbol('C1')
        self.C2 = sp.Symbol('C2')
        self.transformations = (standard_transformations + (implicit_multiplication, convert_xor))

    def resolver_no_homogeneas(self, data):
        """Resuelve ecuaciones diferenciales no homogéneas con coeficientes constantes"""
        try:
            ecuacion_str = data.get('equation', '').strip()
            condiciones = data.get('conditions', '').strip()
            
            if not ecuacion_str:
                return {
                    'status': 'error',
                    'message': 'La ecuación no puede estar vacía'
                }
            
            ecuacion_str = self._preprocesar_ecuacion(ecuacion_str)
            expr = self._parsear_ecuacion(ecuacion_str)
            orden = self._determinar_orden(expr)
            
            if not self._es_lineal(expr):
                return {
                    'status': 'error',
                    'message': 'La ecuación debe ser lineal'
                }
            
            if self._es_homogenea(expr):
                return {
                    'status': 'error',
                    'message': 'Esta es una ecuación homogénea. Use el resolvedor de ecuaciones homogéneas.'
                }
            
            if orden == 1:
                return self._resolver_orden_1_no_homogenea(expr, condiciones)
            elif orden == 2:
                return self._resolver_orden_2_no_homogenea(expr, condiciones)
            else:
                return {
                    'status': 'error',
                    'message': f'Solo se soportan ecuaciones de orden 1 y 2, esta es de orden {orden}'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error al resolver la ecuación: {str(e)}',
                'traceback': traceback.format_exc()
            }

    def _resolver_orden_1_no_homogenea(self, expr, condiciones):
        """Resuelve ecuaciones no homogéneas de primer orden: y' + P(x)y = Q(x)"""
        x = self.x
        y = self.y
        steps = []
        particular_solution = None
        additional_steps = []
        graph_data = None

        try:
            # Paso 1: Identificar P(x) y Q(x) en la forma y' + P(x)y = Q(x)
            dy_dx = next(d for d in expr.atoms(sp.Derivative) 
                     if d.expr == y and self._get_derivative_order(d) == 1)
            
            # Normalizar la ecuación
            coef_dy = expr.coeff(dy_dx)
            if coef_dy and coef_dy != 1:
                expr = expr / coef_dy
            
            P = expr.coeff(y) if expr.coeff(y) is not None else 0
            Q = -(expr - dy_dx - P*y)  # Q(x) es el término no homogéneo
            
            steps.append({
                'step': 1,
                'description': 'Forma estándar de la ecuación',
                'equation': safe_latex(Eq(dy_dx + P*y, Q)),
                'method': 'Identificar P(x) y Q(x)'
            })
            
            # Paso 2: Resolver la ecuación homogénea asociada
            if not P.is_constant():
                return {
                    'status': 'error',
                    'message': 'Solo se soportan ecuaciones con coeficientes constantes'
                }
            
            y_h = self.C1 * sp.exp(-P * x)
            
            steps.append({
                'step': 2,
                'description': 'Solución de la ecuación homogénea',
                'equation': f"y_h = {safe_latex(y_h)}",
                'method': 'Solución estándar para EDO lineal homogénea'
            })
            
            # Paso 3: Encontrar solución particular
            y_p = self._encontrar_solucion_particular_orden1(P, Q)
            
            steps.append({
                'step': 3,
                'description': 'Solución particular',
                'equation': f"y_p = {safe_latex(y_p)}",
                'method': 'Método de coeficientes indeterminados'
            })
            
            # Paso 4: Solución general
            y_func = sp.Function('y')(x)
            solucion_general = Eq(y_func, y_h + y_p)
            
            steps.append({
                'step': 4,
                'description': 'Solución general',
                'equation': safe_latex(solucion_general),
                'method': 'y = y_h + y_p'
            })
            
            # Procesar condiciones iniciales
            if condiciones:
                condiciones_procesadas = self._procesar_condiciones(condiciones)
                if condiciones_procesadas:
                    tipo_cond, x0, y0 = condiciones_procesadas[0]
                    
                    if tipo_cond == 'valor':
                        x0 = self._format_number(x0)
                        y0 = self._format_number(y0)
                        ecuacion_condicion = solucion_general.rhs.subs(x, x0) - y0
                        
                        additional_steps.append({
                            'step': 1,
                            'description': f'Aplicación condición y({x0}) = {y0}',
                            'equation': safe_latex(Eq(solucion_general.rhs.subs(x, x0), y0)),
                            'method': 'Sustitución directa'
                        })
                        
                        valor_C1 = sp.solve(ecuacion_condicion, self.C1, rational=True)
                        
                        if valor_C1:
                            c1_val = valor_C1[0]
                            c1_val = sp.simplify(c1_val)
                            
                            additional_steps.append({
                                'step': 2,
                                'description': 'Resolver para C1',
                                'equation': f"C1 = {safe_latex(c1_val)}",
                                'method': 'Despeje algebraico'
                            })
                            
                            solucion_particular = Eq(y_func, solucion_general.rhs.subs(self.C1, c1_val))
                            particular_solution = safe_latex(solucion_particular)
                            
                            additional_steps.append({
                                'step': 3,
                                'description': 'Solución particular',
                                'equation': particular_solution,
                                'method': 'Sustitución de constante'
                            })
                            
                            # Generar gráfico
                            graph_data = self._generar_datos_matplotlib(solucion_particular.rhs)

            return {
                'status': 'success',
                'solution': safe_latex(solucion_general),
                'particular_solution': particular_solution,
                'steps': steps,
                'additional_steps': additional_steps,
                'graph_data': graph_data,
                'needs_conditions': not bool(condiciones)
            }

        except Exception as e:
            print(f"Error en _resolver_orden_1_no_homogenea: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'steps': steps,
                'traceback': traceback.format_exc()
            }

    def _resolver_orden_2_no_homogenea(self, expr, condiciones):
        """Resuelve ecuaciones no homogéneas de segundo orden: y'' + ay' + by = f(x)"""
        x = self.x
        y = self.y
        steps = []
        
        try:
            # Paso 1: Normalizar la ecuación
            dy_dx2 = next(d for d in expr.atoms(sp.Derivative) 
                         if d.expr == y and self._get_derivative_order(d) == 2)
            
            coef_y_der2 = expr.coeff(dy_dx2)
            expr_normalizado = expr / coef_y_der2 if coef_y_der2 != 1 else expr
            
            a = expr_normalizado.coeff(sp.Derivative(y, x))
            b = expr_normalizado.coeff(y)
            
            # Extraer el término no homogéneo
            termino_homogeneo = dy_dx2 + (a if a else 0)*sp.Derivative(y, x) + (b if b else 0)*y
            f_x = -(expr_normalizado - termino_homogeneo)
            
            steps.append({
                'step': 1,
                'description': 'Forma estándar de la ecuación',
                'equation': safe_latex(Eq(termino_homogeneo, f_x)),
                'method': 'Normalización y extracción de f(x)'
            })
            
            # Paso 2: Resolver la ecuación homogénea asociada
            r = sp.Symbol('r')
            ecuacion_caracteristica = Eq(r**2 + (a if a else 0)*r + (b if b else 0), 0)
            
            steps.append({
                'step': 2,
                'description': 'Ecuación característica',
                'equation': safe_latex(ecuacion_caracteristica),
                'method': 'Reemplazar y por e^(rx)'
            })
            
            raices = sp.solve(ecuacion_caracteristica, r)
            
            if len(raices) == 2:
                if raices[0] == raices[1]:
                    y_h = (self.C1 + self.C2 * x) * sp.exp(raices[0] * x)
                    tipo_raices = "Raíz real doble"
                elif all(r.is_real for r in raices):
                    y_h = self.C1 * sp.exp(raices[0] * x) + self.C2 * sp.exp(raices[1] * x)
                    tipo_raices = "Dos raíces reales distintas"
                else:
                    alpha = sp.re(raices[0])
                    beta = sp.im(raices[0])
                    y_h = sp.exp(alpha * x) * (self.C1 * sp.cos(beta * x) + self.C2 * sp.sin(beta * x))
                    tipo_raices = "Raíces complejas conjugadas"
            else:
                raise ValueError("La ecuación característica no produjo 2 raíces")
            
            steps.append({
                'step': 3,
                'description': 'Solución homogénea',
                'equation': f"y_h = {safe_latex(y_h)}",
                'details': f"Tipo: {tipo_raices}",
                'method': 'Solución según tipo de raíces'
            })
            
            # Paso 3: Encontrar solución particular
            y_p = self._encontrar_solucion_particular_orden2(a if a else 0, b if b else 0, f_x, y_h)
            
            steps.append({
                'step': 4,
                'description': 'Solución particular',
                'equation': f"y_p = {safe_latex(y_p)}",
                'method': 'Método de coeficientes indeterminados'
            })
            
            # Paso 4: Solución general
            y_func = sp.Function('y')(x)
            solucion_general = Eq(y_func, y_h + y_p)
            
            steps.append({
                'step': 5,
                'description': 'Solución general',
                'equation': safe_latex(solucion_general),
                'method': 'y = y_h + y_p'
            })
            
            particular_solution = None
            additional_steps = []
            graph_data = None
            
            # Procesar condiciones iniciales
            if condiciones:
                try:
                    condiciones_procesadas = self._procesar_condiciones(condiciones)
                    if len(condiciones_procesadas) >= 2:
                        for i, cond in enumerate(condiciones_procesadas):
                            tipo, x_val, y_val = cond
                            x_val = self._format_number(x_val)
                            y_val = self._format_number(y_val)
                            condiciones_procesadas[i] = (tipo, x_val, y_val)
                        
                        cond1 = [c for c in condiciones_procesadas if c[0] == 'valor'][0]
                        eq1 = solucion_general.subs({x: cond1[1], y: cond1[2]})
                        
                        y_der = sp.diff(solucion_general.rhs, x)
                        cond2 = [c for c in condiciones_procesadas if c[0] == 'derivada'][0]
                        eq2 = Eq(y_der.subs(x, cond2[1]), cond2[2])
                        
                        sol_constantes = sp.solve((eq1, eq2), (self.C1, self.C2), rational=True)
                        if sol_constantes:
                            sol_constantes = {k: sp.simplify(v) for k, v in sol_constantes.items()}
                            solucion_particular = solucion_general.subs(sol_constantes)
                            particular_solution = safe_latex(solucion_particular)
                            
                            additional_steps.extend([
                                {
                                    'step': 1,
                                    'description': f'Aplicación condición y({cond1[1]}) = {cond1[2]}',
                                    'equation': safe_latex(eq1)
                                },
                                {
                                    'step': 2,
                                    'description': f'Aplicación condición y\'({cond2[1]}) = {cond2[2]}',
                                    'equation': safe_latex(eq2)
                                },
                                {
                                    'step': 3,
                                    'description': 'Solución para constantes',
                                    'equation': f"C1 = {safe_latex(sol_constantes[self.C1])}, C2 = {safe_latex(sol_constantes[self.C2])}"
                                },
                                {
                                    'step': 4,
                                    'description': 'Solución particular',
                                    'equation': particular_solution
                                }
                            ])
                            
                            # Generar gráfico
                            graph_data = self._generar_datos_matplotlib(solucion_particular.rhs)
                            
                except Exception as e:
                    print(f"Error procesando condiciones orden 2: {str(e)}")
                    additional_steps.append({
                        'step': len(additional_steps)+1,
                        'description': 'Error aplicando condiciones',
                        'error': str(e)
                    })
            
            return {
                'status': 'success',
                'solution': safe_latex(solucion_general),
                'particular_solution': particular_solution,
                'steps': steps,
                'additional_steps': additional_steps,
                'graph_data': graph_data,
                'roots_info': {
                    'roots': [safe_latex(r) for r in raices],
                    'type': tipo_raices
                },
                'needs_conditions': not bool(condiciones)
            }
            
        except Exception as e:
            steps.append({
                'step': len(steps)+1,
                'description': 'Error en el proceso',
                'error': str(e)
            })
            return {
                'status': 'error',
                'message': str(e),
                'steps': steps,
                'traceback': traceback.format_exc()
            }

    def _encontrar_solucion_particular_orden1(self, P, Q):
        """Encuentra solución particular para ecuaciones de primer orden"""
        x = self.x
        
        # Analizar el tipo de Q(x)
        if Q == 0:
            return 0
        
        # Caso 1: Q es constante
        if Q.is_constant():
            return Q / P if P != 0 else None
        
        # Caso 2: Q es exponencial
        if Q.has(sp.exp):
            # Extraer el argumento de la exponencial
            exp_terms = [arg for arg in Q.atoms(sp.exp)]
            if exp_terms:
                exp_arg = exp_terms[0].args[0]
                k = exp_arg.coeff(x) if x in exp_arg.free_symbols else 0
                
                # Si k = -P, necesitamos multiplicar por x
                if k == -P:
                    A = sp.Symbol('A')
                    y_p_propuesta = A * x * sp.exp(k * x)
                else:
                    A = sp.Symbol('A')
                    y_p_propuesta = A * sp.exp(k * x)
                
                # Sustituir en la ecuación
                dy_p = sp.diff(y_p_propuesta, x)
                ecuacion = dy_p + P * y_p_propuesta - Q
                A_val = sp.solve(ecuacion, A)[0]
                return y_p_propuesta.subs(A, A_val)
        
        # Caso 3: Q es polinomio
        if Q.is_polynomial(x):
            grado = sp.degree(Q, x)
            coeffs = [sp.Symbol(f'A{i}') for i in range(grado + 1)]
            y_p_propuesta = sum(coeffs[i] * x**i for i in range(grado + 1))
            
            dy_p = sp.diff(y_p_propuesta, x)
            ecuacion = dy_p + P * y_p_propuesta - Q
            
            # Resolver para los coeficientes
            eq_expandida = sp.expand(ecuacion)
            sistema = []
            for i in range(grado + 1):
                coef = eq_expandida.coeff(x, i)
                if coef != 0:
                    sistema.append(coef)
            
            sol = sp.solve(sistema, coeffs)
            return y_p_propuesta.subs(sol)
        
        # Caso 4: Q es trigonométrico
        if Q.has(sp.sin) or Q.has(sp.cos):
            # Extraer frecuencia
            for func in [sp.sin, sp.cos]:
                if Q.has(func):
                    trig_terms = [arg for arg in Q.atoms(func)]
                    if trig_terms:
                        omega = trig_terms[0].args[0].coeff(x) if x in trig_terms[0].args[0].free_symbols else 0
                        break
            
            A = sp.Symbol('A')
            B = sp.Symbol('B')
            y_p_propuesta = A * sp.sin(omega * x) + B * sp.cos(omega * x)
            
            dy_p = sp.diff(y_p_propuesta, x)
            ecuacion = dy_p + P * y_p_propuesta - Q
            
            # Resolver para A y B
            eq_expandida = sp.expand(ecuacion)
            eq_sin = eq_expandida.coeff(sp.sin(omega * x))
            eq_cos = eq_expandida.coeff(sp.cos(omega * x))
            
            sol = sp.solve([eq_sin, eq_cos], [A, B])
            return y_p_propuesta.subs(sol)
        
        # Si no se puede determinar, usar variación de parámetros
        return self._variacion_parametros_orden1(P, Q)

    def _encontrar_solucion_particular_orden2(self, a, b, f_x, y_h):
        """Encuentra solución particular para ecuaciones de segundo orden"""
        x = self.x
        
        if f_x == 0:
            return 0
        
        # Determinar el tipo de f(x) y proponer solución particular
        y_p = 0
        
        # Caso 1: f(x) es exponencial
        if f_x.has(sp.exp):
            exp_terms = list(f_x.atoms(sp.exp))
            for exp_term in exp_terms:
                if exp_term.has(x):
                    k = exp_term.args[0].coeff(x) if x in exp_term.args[0].free_symbols else 0
                    coef = f_x.coeff(exp_term)
                    
                    # Verificar si e^(kx) está en y_h
                    s = 0
                    if sp.exp(k*x) in y_h.atoms(sp.exp):
                        s = 1
                        if x * sp.exp(k*x) in y_h.atoms(sp.Mul):
                            s = 2
                    
                    A = sp.Symbol('A')
                    y_p_term = A * x**s * sp.exp(k * x)
                    
                    # Sustituir y resolver
                    y_p_term_final = self._resolver_coeficiente(y_p_term, a, b, coef * exp_term, A)
                    y_p += y_p_term_final
        
        # Caso 2: f(x) es polinomio
        elif f_x.is_polynomial(x):
            n = sp.degree(f_x, x)
            s = 0
            
            # Verificar si necesitamos multiplicar por x
            if b == 0:
                if a == 0:
                    s = 2
                else:
                    s = 1
            
            coeffs = [sp.Symbol(f'A{i}') for i in range(n + s + 1)]
            y_p = sum(coeffs[i] * x**(i) for i in range(n + s + 1))
            
            # Sustituir y resolver
            dy_p = sp.diff(y_p, x)
            d2y_p = sp.diff(y_p, x, 2)
            ecuacion = d2y_p + a * dy_p + b * y_p - f_x
            
            # Resolver sistema
            eq_expandida = sp.expand(ecuacion)
            sistema = []
            for i in range(n + s + 1):
                coef = eq_expandida.coeff(x, i)
                if coef != 0:
                    sistema.append(coef)
            
            sol = sp.solve(sistema, coeffs)
            y_p = y_p.subs(sol)
        
        # Caso 3: f(x) es trigonométrico
        elif f_x.has(sp.sin) or f_x.has(sp.cos):
            # Extraer frecuencia omega
            omega = None
            for func in [sp.sin, sp.cos]:
                if f_x.has(func):
                    trig_terms = list(f_x.atoms(func))
                    if trig_terms:
                        arg = trig_terms[0].args[0]
                        omega = arg.coeff(x) if x in arg.free_symbols else arg
                        break
            
            if omega is not None:
                A = sp.Symbol('A')
                B = sp.Symbol('B')
                
                # Verificar si sin(omega*x) o cos(omega*x) están en y_h
                s = 0
                if (sp.sin(omega*x) in y_h.atoms(sp.sin) or 
                    sp.cos(omega*x) in y_h.atoms(sp.cos)):
                    s = 1
                
                y_p = x**s * (A * sp.sin(omega * x) + B * sp.cos(omega * x))
                
                # Sustituir y resolver
                dy_p = sp.diff(y_p, x)
                d2y_p = sp.diff(y_p, x, 2)
                ecuacion = d2y_p + a * dy_p + b * y_p - f_x
                
                # Expandir y extraer coeficientes
                eq_expandida = sp.expand(ecuacion)
                eq_sin = eq_expandida.coeff(sp.sin(omega * x))
                eq_cos = eq_expandida.coeff(sp.cos(omega * x))
                
                # Si uno de los coeficientes es None, intentar con trigsimp
                if eq_sin is None or eq_cos is None:
                    eq_expandida = sp.trigsimp(ecuacion)
                    eq_sin = eq_expandida.coeff(sp.sin(omega * x))
                    eq_cos = eq_expandida.coeff(sp.cos(omega * x))
                
                sol = sp.solve([eq_sin, eq_cos], [A, B])
                y_p = y_p.subs(sol)
        
        # Caso 4: Combinación de términos
        elif f_x.is_Add:
            for termino in f_x.args:
                y_p += self._encontrar_solucion_particular_orden2(a, b, termino, y_h)
        
        return sp.simplify(y_p)

    def _resolver_coeficiente(self, y_p_propuesta, a, b, termino, coef_symbol):
        """Resuelve para el coeficiente en la solución particular propuesta"""
        x = self.x
        
        dy_p = sp.diff(y_p_propuesta, x)
        d2y_p = sp.diff(y_p_propuesta, x, 2)
        
        ecuacion = d2y_p + a * dy_p + b * y_p_propuesta - termino
        
        # Resolver para el coeficiente
        sol = sp.solve(ecuacion, coef_symbol)
        if sol:
            return y_p_propuesta.subs(coef_symbol, sol[0])
        return 0

    def _variacion_parametros_orden1(self, P, Q):
        """Método de variación de parámetros para primer orden"""
        x = self.x
        
        # Factor integrante
        mu = sp.exp(sp.integrate(P, x))
        
        # Integral particular
        integral = sp.integrate(mu * Q, x)
        
        return integral / mu

    def _generar_datos_matplotlib(self, solucion_expr):
        """Genera datos para graficar la solución con matplotlib"""
        try:
            if not solucion_expr:
                return None

            # Convertir la expresión sympy a una función numérica
            f = lambdify(self.x, solucion_expr, modules=['numpy', 'sympy'])
            
            # Determinar un rango de x adecuado
            x_vals = np.linspace(-5, 5, 500)
            
            # Evaluar la función
            y_vals = []
            x_valid = []
            
            for x_val in x_vals:
                try:
                    y_val = complex(f(x_val))
                    if np.isfinite(y_val) and abs(y_val.imag) < 1e-10:
                        y_vals.append(float(y_val.real))
                        x_valid.append(float(x_val))
                except:
                    continue
            
            if len(y_vals) >= 10:
                # Crear la gráfica
                plt.figure(figsize=(10, 6))
                plt.plot(x_valid, y_vals, 'r-', linewidth=2, label='Solución Particular')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title('Solución de la Ecuación Diferencial No Homogénea')
                plt.grid(True)
                plt.legend()
                
                # Guardar en buffer
                buf = BytesIO()
                plt.savefig(buf, format='png', dpi=100)
                plt.close()
                
                # Codificar en base64
                image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                buf.close()
                
                return {
                    'image': f"data:image/png;base64,{image_base64}",
                    'x_values': x_valid,
                    'y_values': y_vals
                }
            
            return None
            
        except Exception as e:
            print(f"Error en _generar_datos_matplotlib: {str(e)}")
            return None

    def _preprocesar_ecuacion(self, ecuacion_str):
        """Preprocesa la ecuación para facilitar el parsing"""
        ecuacion_str = ecuacion_str.replace("^", "**")  # Corregido: mantener ^ como **
        
        ecuacion_str = re.sub(r"y\s*''''", "Derivative(y, (x, 4))", ecuacion_str)
        ecuacion_str = re.sub(r"y\s*'''", "Derivative(y, (x, 3))", ecuacion_str)
        ecuacion_str = re.sub(r"y\s*''", "Derivative(y, (x, 2))", ecuacion_str)
        ecuacion_str = re.sub(r"y\s*'", "Derivative(y, x)", ecuacion_str)
        ecuacion_str = re.sub(r"y\s*\(\s*(\d+)\s*\)", r"Derivative(y, (x, \1))", ecuacion_str)
        ecuacion_str = re.sub(r"d\s*(\d*)\s*y\s*/\s*d\s*x\s*\^?\s*(\d*)", self._reemplazar_derivada, ecuacion_str, flags=re.IGNORECASE)
        
        return ecuacion_str

    def _reemplazar_derivada(self, match):
        """Reemplaza notación dy/dx por Derivative"""
        orden = match.group(2) or (match.group(1) if match.group(1) else '1')
        return f"Derivative(y, (x, {orden}))"

    def _parsear_ecuacion(self, ecuacion_str):
        """Parsea la ecuación y la convierte a expresión sympy"""
        try:
            if "=" not in ecuacion_str:
                ecuacion_str = f"{ecuacion_str} = 0"  # Permitir ecuaciones sin =0
                
            lhs, rhs = ecuacion_str.split("=", 1)
            
            # Crear función y localmente para el parsing
            y = self.y
            
            expr = parse_expr(f"{lhs.strip()} - ({rhs.strip()})", 
                            local_dict={'y': y, 'x': self.x},
                            transformations=self.transformations)
            
            if not (expr.has(sp.Derivative) or expr.has(y)):
                raise ValueError("La ecuación debe contener al menos una función y y su derivada")
                
            return expr
        except Exception as e:
            raise ValueError(f"Error al parsear la ecuación: {str(e)}")

    def _determinar_orden(self, expr):
        """Determina el orden de la ecuación diferencial"""
        max_order = 0
        y = self.y
        for der in expr.atoms(sp.Derivative):
            if der.expr == y:
                if len(der.variables) == 1 and isinstance(der.variables[0], tuple):
                    order = der.variables[0][1]
                else:
                    order = len(der.variables)
                max_order = max(max_order, order)
        return max_order

    def _es_lineal(self, expr):
        """Verifica si la ecuación es lineal"""
        y = self.y
        derivadas = [d for d in expr.atoms(sp.Derivative) if d.expr == y]
        
        for term in sp.Add.make_args(expr):
            coef = term
            for der in sorted(derivadas, key=lambda d: -self._get_derivative_order(d)):
                if der in term.atoms():
                    coef = term.coeff(der)
                    break
            else:
                if y in term.atoms():
                    coef = term.coeff(y)
                else:
                    continue
            
            if coef and any(sym == y or isinstance(sym, sp.Derivative) for sym in coef.free_symbols):
                return False
            
            if term.is_Mul:
                vars_in_term = sum(1 for arg in term.args 
                                 if arg == y or any(arg == d for d in derivadas))
                if vars_in_term > 1:
                    return False
        
        return True

    def _get_derivative_order(self, der):
        """Obtiene el orden de una derivada"""
        if hasattr(der, 'derivative_count'):
            return der.derivative_count
        elif len(der.variables) == 1:
            if isinstance(der.variables[0], tuple):
                return der.variables[0][1]
            else:
                return 1
        else:
            return len(der.variables)

    def _es_homogenea(self, expr):
        """Determina si la ecuación es homogénea"""
        if not self._es_lineal(expr):
            return False
        
        y = self.y    
        # Una ecuación es no homogénea si tiene términos que no contienen y o sus derivadas
        for term in sp.Add.make_args(expr):
            tiene_y = False
            
            # Verificar si el término contiene y o alguna de sus derivadas
            if term.has(y):
                tiene_y = True
            else:
                for atom in term.atoms():
                    if isinstance(atom, sp.Derivative) and atom.expr == y:
                        tiene_y = True
                        break
            
            # Si el término no contiene y ni sus derivadas, la ecuación es no homogénea
            if not tiene_y:
                return False
        
        return True

    def _procesar_condiciones(self, condiciones_str):
        """Procesa las condiciones iniciales"""
        condiciones = []
        if not condiciones_str:
            return condiciones
        
        for cond in condiciones_str.split(','):
            cond = cond.strip()
            if not cond:
                continue
            
            # Condición para y(x0) = y0
            match = re.match(r"y\(([^)]+)\)\s*=\s*([^ ]+)", cond)
            if match:
                try:
                    x0 = float(match.group(1))
                    y0 = float(match.group(2))
                    condiciones.append(('valor', x0, y0))
                    continue
                except ValueError:
                    pass
            
            # Condición para y'(x0) = y0
            match = re.match(r"y\'\(([^)]+)\)\s*=\s*([^ ]+)", cond)
            if match:
                try:
                    x0 = float(match.group(1))
                    y0 = float(match.group(2))
                    condiciones.append(('derivada', x0, y0))
                    continue
                except ValueError:
                    pass
                    
        return condiciones

    def _format_number(self, num):
        """Formatea números para mantener fracciones cuando sea posible"""
        if isinstance(num, (int, float)):
            if float(num).is_integer():
                return int(num)
            return sp.Rational(num).limit_denominator()
        return num

def resolver_no_homogeneas(data):
    """Función principal para resolver ecuaciones no homogéneas"""
    resolvedor = ResolvedorEcuacionesNoHomogeneas()
    return resolvedor.resolver_no_homogeneas(data)