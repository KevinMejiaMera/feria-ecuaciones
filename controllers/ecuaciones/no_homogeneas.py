import re
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication, convert_xor
import traceback
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sympy import Eq, lambdify
import numpy as np

class ResolvedorEcuacionesNoHomogeneas:
    def __init__(self):
        self.x = sp.Symbol('x')
        self.y = sp.Function('y')(self.x)
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
            
            try:
                expr = self._parsear_ecuacion(ecuacion_str)
            except Exception as e:
                return {
                    'status': 'error',
                    'message': f'Error al parsear la ecuación: {str(e)}',
                    'sugerencia': 'Revise la sintaxis. Use y\' para primera derivada, y\'\' para segunda, etc.'
                }
            
            if not (expr.has(sp.Derivative) or expr.has(self.y)):
                return {
                    'status': 'error',
                    'message': 'La ecuación debe contener la función y y su derivada'
                }
            
            orden = self._determinar_orden(expr)
            
            if orden not in [1, 2]:
                return {
                    'status': 'error',
                    'message': f'Solo se soportan ecuaciones de orden 1 y 2. Esta es de orden {orden}'
                }
            
            if not self._es_lineal(expr):
                return {
                    'status': 'error',
                    'message': 'La ecuación debe ser lineal',
                    'sugerencia': 'No se soportan términos no lineales como y*y\' o y^2'
                }
            
            if self._es_homogenea(expr) == "Homogénea":
                return {
                    'status': 'error',
                    'message': 'La ecuación parece ser homogénea (igual a cero)',
                    'sugerencia': 'Use el resolvedor de ecuaciones homogéneas para este caso'
                }
            
            if orden == 1:
                return self._resolver_orden_1_no_homogenea(expr, condiciones)
            elif orden == 2:
                return self._resolver_orden_2_no_homogenea(expr, condiciones)
                
        except Exception as e:
            traceback.print_exc()
            return {
                'status': 'error',
                'message': f'Error inesperado al resolver la ecuación: {str(e)}',
                'traceback': traceback.format_exc()
            }

    def _format_number(self, num):
        """Formatea números para mantener fracciones cuando sea posible"""
        if isinstance(num, (int, float)):
            if num.is_integer():
                return int(num)
            return sp.Rational(num).limit_denominator()
        return num
    
    def _resolver_orden_1_no_homogenea(self, expr, condiciones):
        x, y = self.x, self.y
        steps = []
        particular_solution = None
        additional_steps = []

        try:
            # Paso 1: Identificar ecuación homogénea asociada y' + P(x)y = 0
            P = expr.coeff(y)
            if P is None:
                P = 0
            term_homogeneo = sp.Derivative(y, x) + P*y
            term_no_homogeneo = expr - term_homogeneo
            
            if term_no_homogeneo == 0:
                return {
                    'status': 'error',
                    'message': 'La ecuación parece ser homogénea (término no homogéneo = 0)'
                }
            
            steps.append({
                'step': 1,
                'description': 'Identificar ecuación homogénea asociada',
                'equation': f"Ecuación homogénea: {sp.latex(Eq(term_homogeneo, 0))}",
                'method': 'Forma estándar y\' + P(x)y = 0'
            })
            
            # Paso 2: Resolver la homogénea y_h = C1*e^(-∫P dx)
            try:
                if not P.is_constant():
                    raise ValueError("Solo se soportan coeficientes constantes")
                
                y_h = self.C1 * sp.exp(-P*x)
                
                steps.append({
                    'step': 2,
                    'description': 'Solución de la ecuación homogénea',
                    'equation': f"y_h = {sp.latex(y_h)}",
                    'method': 'Solución para ecuaciones lineales de primer orden'
                })
            except Exception as e:
                return {
                    'status': 'error',
                    'message': f'Error al resolver la parte homogénea: {str(e)}',
                    'steps': steps
                }
            
            # Paso 3: Encontrar solución particular
            try:
                y_p = self._encontrar_solucion_particular(term_homogeneo, term_no_homogeneo, orden=1)
                
                steps.append({
                    'step': 3,
                    'description': 'Solución particular encontrada',
                    'equation': f"y_p = {sp.latex(y_p)}",
                    'method': 'Coeficientes indeterminados'
                })
            except Exception as e:
                return {
                    'status': 'error',
                    'message': f'Error al encontrar solución particular: {str(e)}',
                    'steps': steps
                }
            
            # Paso 4: Solución general y = y_h + y_p
            solucion_general = Eq(y, y_h + y_p)
            
            steps.append({
                'step': 4,
                'description': 'Solución general (homogénea + particular)',
                'equation': sp.latex(solucion_general),
                'method': 'Principio de superposición'
            })
            
            # Aplicar condiciones iniciales si existen
            if condiciones:
                condiciones_procesadas = self._procesar_condiciones(condiciones)
                if condiciones_procesadas:
                    tipo_cond, x0, y0 = condiciones_procesadas[0]
                    x0 = self._format_number(x0)
                    y0 = self._format_number(y0)
                    
                    ecuacion_condicion = solucion_general.rhs.subs(x, x0) - y0
                    
                    additional_steps.append({
                        'step': 1,
                        'description': f'Aplicación condición y({x0}) = {y0}',
                        'equation': sp.latex(Eq(solucion_general.rhs.subs(x, x0), y0)),
                        'method': 'Sustitución directa'
                    })
                    
                    valor_C1 = sp.solve(ecuacion_condicion, self.C1, rational=True)
                    
                    if valor_C1:
                        c1_val = valor_C1[0]
                        c1_val = sp.simplify(c1_val)
                        
                        additional_steps.append({
                            'step': 2,
                            'description': 'Resolver para C1',
                            'equation': f"C1 = {sp.latex(c1_val)}",
                            'method': 'Despeje algebraico'
                        })
                        
                        solucion_particular = Eq(y, c1_val * sp.exp(-P*x) + y_p)
                        particular_solution = sp.latex(solucion_particular)
                        
                        additional_steps.append({
                            'step': 3,
                            'description': 'Solución particular',
                            'equation': particular_solution,
                            'method': 'Sustitución de constante'
                        })

            plot_html = self._generar_grafico_matplotlib(solucion_general, particular_solution)
        
            return {
                'status': 'success',
                'solution': sp.latex(solucion_general),
                'particular_solution': particular_solution,
                'steps': steps,
                'additional_steps': additional_steps,
                'plot_html': plot_html,
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
        x, y = self.x, self.y
        steps = []
        
        try:
            # Paso 1: Forma estándar y'' + a y' + b y = f(x)
            dy_dx2 = next(d for d in expr.atoms(sp.Derivative) 
                     if d.expr == y and self._get_derivative_order(d) == 2)
            
            coef_y_der2 = expr.coeff(dy_dx2)
            expr_normalizado = expr / coef_y_der2 if coef_y_der2 != 1 else expr
            
            a = expr_normalizado.coeff(sp.Derivative(y, x))
            b = expr_normalizado.coeff(y)
            term_no_homogeneo = expr_normalizado - (sp.Derivative(y, (x, 2)) + a*sp.Derivative(y, x) + b*y)
            
            steps.append({
                'step': 1,
                'description': 'Ecuación en forma estándar',
                'equation': f"y'' + {sp.latex(a)}y' + {sp.latex(b)}y = {sp.latex(term_no_homogeneo)}",
                'method': 'Normalización de la ecuación'
            })
            
            # Paso 2: Resolver la homogénea
            try:
                r = sp.Symbol('r')
                ecuacion_caracteristica = Eq(r**2 + a*r + b, 0)
                raices = sp.roots(ecuacion_caracteristica, r, multiple=True)
                
                if len(raices) < 2:
                    if len(raices) == 1:
                        raices = [raices[0], raices[0]]
                    else:
                        raise ValueError("No se pudieron encontrar raíces características")
                
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
                
                steps.append({
                    'step': 2,
                    'description': 'Solución de la ecuación homogénea',
                    'equation': f"y_h = {sp.latex(y_h)}",
                    'method': 'Método para ecuaciones homogéneas',
                    'roots_info': {
                        'roots': [sp.latex(r) for r in raices],
                        'type': tipo_raices
                    }
                })
            except Exception as e:
                return {
                    'status': 'error',
                    'message': f'Error al resolver la parte homogénea: {str(e)}',
                    'steps': steps
                }
            
            # Paso 3: Encontrar solución particular
            try:
                y_p = self._encontrar_solucion_particular(sp.Derivative(y, (x, 2)) + a*sp.Derivative(y, x) + b*y, 
                                                         term_no_homogeneo, orden=2)
                
                steps.append({
                    'step': 3,
                    'description': 'Solución particular encontrada',
                    'equation': f"y_p = {sp.latex(y_p)}",
                    'method': 'Coeficientes indeterminados'
                })
            except Exception as e:
                return {
                    'status': 'error',
                    'message': f'Error al encontrar solución particular: {str(e)}',
                    'steps': steps
                }
            
            # Paso 4: Solución general
            solucion_general = Eq(y, y_h + y_p)
            
            steps.append({
                'step': 4,
                'description': 'Solución general (homogénea + particular)',
                'equation': sp.latex(solucion_general),
                'method': 'Principio de superposición'
            })
            
            # Aplicar condiciones iniciales
            particular_solution = None
            additional_steps = []
            
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
                            particular_solution = sp.latex(solucion_particular)
                            
                            additional_steps.extend([
                                {
                                    'step': 1,
                                    'description': f'Aplicación condición y({cond1[1]}) = {cond1[2]}',
                                    'equation': sp.latex(eq1)
                                },
                                {
                                    'step': 2,
                                    'description': f'Aplicación condición y\'({cond2[1]}) = {cond2[2]}',
                                    'equation': sp.latex(eq2)
                                },
                                {
                                    'step': 3,
                                    'description': 'Solución para constantes',
                                    'equation': f"C1 = {sp.latex(sol_constantes[self.C1])}, C2 = {sp.latex(sol_constantes[self.C2])}"
                                },
                                {
                                    'step': 4,
                                    'description': 'Solución particular',
                                    'solution': f"y = {particular_solution}"
                                }
                            ])
                except Exception as e:
                    additional_steps.append({
                        'step': len(additional_steps)+1,
                        'description': 'Error aplicando condiciones',
                        'error': str(e)
                    })
            
            plot_html = self._generar_grafico_matplotlib(solucion_general, particular_solution)
            
            return {
                'status': 'success',
                'solution': sp.latex(solucion_general),
                'particular_solution': particular_solution,
                'steps': steps,
                'additional_steps': additional_steps,
                'plot_html': plot_html,
                'roots_info': {
                    'roots': [sp.latex(r) for r in raices],
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

    def _encontrar_solucion_particular(self, homogenea, no_homogenea, orden):
        """Encuentra una solución particular usando coeficientes indeterminados"""
        x, y = self.x, self.y
        term = no_homogenea
        
        # Obtener solución homogénea para verificar superposición
        if orden == 1:
            sol_homogenea = self._resolver_orden_1(homogenea, "")
        else:
            sol_homogenea = self._resolver_orden_2(homogenea, "")
        
        y_h = parse_expr(sol_homogenea['solution'].split('=')[1].strip(), 
                       transformations=self.transformations)
        
        # Proponer solución particular según el tipo de término
        if term.is_Add:
            terms = term.args
        else:
            terms = [term]
        
        y_p = 0
        for t in terms:
            # Caso exponencial e^kx
            if t.has(sp.exp):
                exp_arg = next(a for a in t.atoms(sp.exp) if a.has(x)).args[0]
                k = exp_arg.coeff(x) if x in exp_arg.free_symbols else 0
                A = sp.Symbol('A')
                
                # Verificar si e^kx está en la solución homogénea
                if any(sp.exp(k*x) in term.args for term in sp.Add.make_args(y_h)):
                    # Si está, multiplicar por x
                    y_p += A * x * sp.exp(k*x)
                else:
                    y_p += A * sp.exp(k*x)
            
            # Caso polinomio
            elif t.is_polynomial(x):
                degree = sp.degree(t, x)
                coeffs = [sp.Symbol(f'A{i}') for i in range(degree + 1)]
                y_p += sum(c * x**i for i, c in enumerate(coeffs))
            
            # Caso trigonométrico a*sin(kx) + b*cos(kx)
            elif t.has(sp.sin) or t.has(sp.cos):
                # Encontrar k para sin(kx) o cos(kx)
                k = None
                for trig in [sp.sin, sp.cos]:
                    if t.has(trig):
                        trig_arg = next(a for a in t.atoms(trig) if a.has(x)).args[0]
                        k = trig_arg.coeff(x) if x in trig_arg.free_symbols else trig_arg
                
                if k is None:
                    raise ValueError("No se pudo determinar el argumento trigonométrico")
                
                A = sp.Symbol('A')
                B = sp.Symbol('B')
                
                # Verificar si sin(kx) o cos(kx) están en la solución homogénea
                if any((sp.sin(k*x) in term.args or sp.cos(k*x) in term.args) 
                   for term in sp.Add.make_args(y_h)):
                    y_p += x * (A * sp.sin(k*x) + B * sp.cos(k*x))
                else:
                    y_p += A * sp.sin(k*x) + B * sp.cos(k*x)
            
            else:
                raise ValueError(f"Tipo de término no soportado: {t}")
        
        # Resolver para los coeficientes
        ecuacion = homogenea.subs(y, y_p)
        for i in range(1, orden + 1):
            derivada = sp.diff(y_p, (x, i))
            ecuacion += homogenea.coeff(sp.Derivative(y, (x, i))) * derivada
        
        ecuacion = sp.expand(ecuacion - no_homogenea)
        coeffs = list(y_p.free_symbols - {x})
        
        # Construir sistema de ecuaciones
        ecuaciones = []
        for potencia in range(sp.degree(ecuacion, x) + 1):
            coef = ecuacion.coeff(x, potencia)
            if coef != 0:
                ecuaciones.append(sp.Eq(coef, 0))
        
        sol_coeffs = sp.solve(ecuaciones, coeffs)
        if not sol_coeffs:
            raise ValueError("No se pudieron determinar los coeficientes")
        
        return y_p.subs(sol_coeffs)

    def _variacion_parametros(self, homogenea, no_homogenea, orden):
        """Método de variación de parámetros para encontrar solución particular"""
        x, y = self.x, self.y
        
        # Primero resolver la homogénea para obtener y1 y y2
        if orden == 1:
            sol_homogenea = self._resolver_orden_1(homogenea, "")
        else:
            sol_homogenea = self._resolver_orden_2(homogenea, "")
        
        y_h = parse_expr(sol_homogenea['solution'].split('=')[1].strip(), 
                       transformations=self.transformations)
        
        if orden == 1:
            y1 = y_h.subs(self.C1, 1)
            W = y1
            u1 = sp.integrate(no_homogenea / y1, x)
            return u1 * y1
        else:
            # Para orden 2, necesitamos dos soluciones independientes
            if self.C1 in y_h.free_symbols and self.C2 in y_h.free_symbols:
                y1 = y_h.subs({self.C1: 1, self.C2: 0})
                y2 = y_h.subs({self.C1: 0, self.C2: 1})
            else:
                # Si no hay dos constantes, asumir que es doble raíz
                y1 = y_h.subs(self.C1, 1)
                y2 = x * y1
            
            # Wronskiano
            W = y1 * sp.diff(y2, x) - y2 * sp.diff(y1, x)
            
            u1 = -sp.integrate(y2 * no_homogenea / W, x)
            u2 = sp.integrate(y1 * no_homogenea / W, x)
            
            return u1 * y1 + u2 * y2

    def _procesar_condiciones(self, condiciones_str):
        condiciones = []
        if not condiciones_str:
            return condiciones
        
        for cond in condiciones_str.split(','):
            cond = cond.strip()
            if not cond:
                continue
            
            match = re.match(r"y\(([^)]+)\)\s*=\s*([^ ]+)", cond)
            if match:
                try:
                    x0 = self._format_number(float(match.group(1)))
                    y0 = self._format_number(float(match.group(2)))
                    condiciones.append(('valor', x0, y0))
                    continue
                except ValueError:
                    pass
            
            match = re.match(r"y\'\(([^)]+)\)\s*=\s*([^ ]+)", cond)
            if match:
                try:
                    x0 = self._format_number(float(match.group(1)))
                    y0 = self._format_number(float(match.group(2)))
                    condiciones.append(('derivada', x0, y0))
                    continue
                except ValueError:
                    pass            
        return condiciones

    def _preprocesar_ecuacion(self, ecuacion_str):
        ecuacion_str = ecuacion_str.replace("^", "**")
        
        ecuacion_str = re.sub(r"y\s*''''", "Derivative(y, (x, 4))", ecuacion_str)
        ecuacion_str = re.sub(r"y\s*'''", "Derivative(y, (x, 3))", ecuacion_str)
        ecuacion_str = re.sub(r"y\s*''", "Derivative(y, (x, 2))", ecuacion_str)
        ecuacion_str = re.sub(r"y\s*'", "Derivative(y, x)", ecuacion_str)
        ecuacion_str = re.sub(r"y\s*\(\s*(\d+)\s*\)", r"Derivative(y, (x, \1))", ecuacion_str)
        ecuacion_str = re.sub(r"d\s*(\d*)\s*y\s*/\s*d\s*x\s*\^?\s*(\d*)", self._reemplazar_derivada, ecuacion_str, flags=re.IGNORECASE)
        
        return ecuacion_str

    def _reemplazar_derivada(self, match):
        orden = match.group(2) or (match.group(1) if match.group(1) else '1')
        return f"Derivative(y, (x, {orden}))"

    def _parsear_ecuacion(self, ecuacion_str):
        try:
            if "=" not in ecuacion_str:
                ecuacion_str = f"{ecuacion_str} = 0"
                
            lhs, rhs = ecuacion_str.split("=", 1)
            expr = parse_expr(f"{lhs.strip()} - ({rhs.strip()})", 
                            local_dict={'y': self.y, 'x': self.x},
                            transformations=self.transformations)
            
            if not (expr.has(sp.Derivative) or expr.has(self.y)):
                raise ValueError("La ecuación debe contener al menos una función y y su derivada")
                
            return expr
        except Exception as e:
            raise ValueError(f"Error al parsear la ecuación: {str(e)}\nEcuación recibida: {ecuacion_str}")

    def _determinar_orden(self, expr):
        max_order = 0
        for der in expr.atoms(sp.Derivative):
            if der.expr == self.y:
                if len(der.variables) == 1 and isinstance(der.variables[0], tuple):
                    order = der.variables[0][1]
                else:
                    order = len(der.variables)
                max_order = max(max_order, order)
        return max_order

    def _es_lineal(self, expr):
        derivadas = [d for d in expr.atoms(sp.Derivative) if d.expr == self.y]
        
        for term in sp.Add.make_args(expr):
            coef = term
            for der in sorted(derivadas, key=lambda d: -self._get_derivative_order(d)):
                if der in term.atoms():
                    coef = term.coeff(der)
                    break
            else:
                if self.y in term.atoms():
                    coef = term.coeff(self.y)
                else:
                    continue
            
            if any(sym == self.y or isinstance(sym, sp.Derivative) for sym in coef.free_symbols):
                return False
            
            if term.is_Mul:
                vars_in_term = sum(1 for arg in term.args 
                                 if arg == self.y or any(arg == d for d in derivadas))
                if vars_in_term > 1:
                    return False
        
        return True

    def _get_derivative_order(self, der):
        if len(der.variables) == 1 and isinstance(der.variables[0], tuple):
            return der.variables[0][1]
        return len(der.variables)

    def _es_homogenea(self, expr):
        if not self._es_lineal(expr):
            return "No homogénea"
            
        for term in sp.Add.make_args(expr):
            if not (term.has(self.y) or any(isinstance(arg, sp.Derivative) for arg in term.atoms())):
                return "No homogénea"
        return "Homogénea"

    def _generar_grafico_matplotlib(self, solucion_general, solucion_particular=None):
        try:
            plt.figure(figsize=(10, 6))
            x_vals = np.linspace(-5, 5, 200)
            
            if isinstance(solucion_general, str):
                expr_general = parse_expr(solucion_general.split('=')[1].strip() if '=' in solucion_general else solucion_general, 
                                       transformations=self.transformations)
            else:
                expr_general = solucion_general.rhs if isinstance(solucion_general, sp.Eq) else solucion_general
            
            if self.C1 in expr_general.free_symbols:
                for c1_val in [-2, -1, 1, 2]:
                    try:
                        expr = expr_general.subs({self.C1: c1_val})
                        f = lambdify(self.x, expr, modules=['numpy'])
                        y_vals = f(x_vals)
                        plt.plot(x_vals, y_vals, '--', label=f'C₁ = {c1_val}')
                    except:
                        continue
            
            if self.C1 in expr_general.free_symbols and self.C2 in expr_general.free_symbols:
                for c1_val, c2_val in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    try:
                        expr = expr_general.subs({self.C1: c1_val, self.C2: c2_val})
                        f = lambdify(self.x, expr, modules=['numpy'])
                        y_vals = f(x_vals)
                        plt.plot(x_vals, y_vals, '--', label=f'C₁ = {c1_val}, C₂ = {c2_val}')
                    except:
                        continue
            
            if solucion_particular:
                try:
                    if isinstance(solucion_particular, str):
                        expr_part = parse_expr(solucion_particular.split('=')[1].strip() if '=' in solucion_particular else solucion_particular,
                                            transformations=self.transformations)
                    else:
                        expr_part = solucion_particular.rhs if isinstance(solucion_particular, sp.Eq) else solucion_particular
                    
                    f = lambdify(self.x, expr_part, modules=['numpy'])
                    y_vals = f(x_vals)
                    plt.plot(x_vals, y_vals, 'r-', linewidth=2, label='Solución particular')
                except Exception as e:
                    print(f"Error procesando solución particular: {str(e)}")
            
            plt.title('Soluciones de la Ecuación Diferencial')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.grid(True)
            plt.legend()
            
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close()
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            
            return f'<img src="data:image/png;base64,{img_base64}" class="img-fluid">'
            
        except Exception as e:
            print(f"Error en _generar_grafico_matplotlib: {str(e)}")
            traceback.print_exc()
            return '<div class="alert alert-danger">Error al generar el gráfico</div>'

    def _resolver_orden_1(self, expr, condiciones):
        """Resuelve ecuaciones homogéneas de primer orden (para uso interno)"""
        x, y = self.x, self.y
        
        P = expr.coeff(y)
        if P is None:
            P = 0
        
        y_func = sp.Function('y')(x)
        solucion_general = Eq(y_func, self.C1 * sp.exp(-P * x))
        
        return {
            'solution': sp.latex(solucion_general),
            'steps': []
        }

    def _resolver_orden_2(self, expr, condiciones):
        """Resuelve ecuaciones homogéneas de segundo orden (para uso interno)"""
        x, y = self.x, self.y
        
        dy_dx2 = next(d for d in expr.atoms(sp.Derivative) 
                 if d.expr == y and self._get_derivative_order(d) == 2)
        
        coef_y_der2 = expr.coeff(dy_dx2)
        expr_normalizado = expr / coef_y_der2 if coef_y_der2 != 1 else expr
        
        a = expr_normalizado.coeff(sp.Derivative(y, x))
        b = expr_normalizado.coeff(y)
        
        r = sp.Symbol('r')
        ecuacion_caracteristica = Eq(r**2 + a*r + b, 0)
        raices = sp.roots(ecuacion_caracteristica, r, multiple=True)
        
        if len(raices) < 2:
            if len(raices) == 1:
                raices = [raices[0], raices[0]]
            else:
                raise ValueError("La ecuación característica no produjo raíces")
        
        if raices[0] == raices[1]:
            solucion_general = Eq(y, (self.C1 + self.C2 * x) * sp.exp(raices[0] * x))
            tipo_raices = "Raíz real doble"
        elif all(r.is_real for r in raices):
            solucion_general = Eq(y, self.C1 * sp.exp(raices[0] * x) + self.C2 * sp.exp(raices[1] * x))
            tipo_raices = "Dos raíces reales distintas"
        else:
            alpha = sp.re(raices[0])
            beta = sp.im(raices[0])
            solucion_general = Eq(y, sp.exp(alpha * x) * 
                          (self.C1 * sp.cos(beta * x) + self.C2 * sp.sin(beta * x)))
            tipo_raices = "Raíces complejas conjugadas"
        
        return {
            'solution': sp.latex(solucion_general),
            'steps': [],
            'roots_info': {
                'roots': [sp.latex(r) for r in raices],
                'type': tipo_raices
            }
        }

def resolver_no_homogeneas(data):
    resolvedor = ResolvedorEcuacionesNoHomogeneas()
    return resolvedor.resolver_no_homogeneas(data)