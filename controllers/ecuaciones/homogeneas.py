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

class ResolvedorEcuacionesHomogeneas:
    def __init__(self):
        self.x = sp.Symbol('x')
        self.y = sp.Function('y')(self.x)
        self.C1 = sp.Symbol('C1')
        self.C2 = sp.Symbol('C2')
        self.transformations = (standard_transformations + (implicit_multiplication, convert_xor))

    def resolver_homogeneas(self, data):
        """Resuelve ecuaciones diferenciales homogéneas con coeficientes constantes"""
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
            
            if not (self._es_lineal(expr) and self._es_homogenea(expr) == "Homogénea"):
                return {
                    'status': 'error',
                    'message': 'La ecuación debe ser lineal homogénea con coeficientes constantes'
                }
            
            if orden == 1:
                return self._resolver_orden_1(expr, condiciones)
            elif orden == 2:
                return self._resolver_orden_2(expr, condiciones)
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

    def _format_number(self, num):
        """Formatea números para mantener fracciones cuando sea posible"""
        if isinstance(num, (int, float)):
            if isinstance(num, float) and num.is_integer():
            return int(num)
        return sp.Rational(num).limit_denominator()
    return num
    
    def _resolver_orden_1(self, expr, condiciones):
        x, y = self.x, self.y
        steps = []
        particular_solution = None
        additional_steps = []
        graph_data = None

        try:
            dy_dx = next(d for d in expr.atoms(sp.Derivative) 
                    if d.expr == y and self._get_derivative_order(d) == 1)
            a = expr.coeff(y) if expr.coeff(y) is not None else 0
        
            r_symbol = sp.Symbol('r')
            ecuacion_caracteristica = Eq(r_symbol + a, 0)
        
            steps.append({
                'step': 1,
                'description': 'Ecuación característica',
                'equation': safe_latex(ecuacion_caracteristica),
                'method': 'Reemplazar y por e^(rx)'
            })
        
            r = sp.solve(ecuacion_caracteristica, r_symbol)[0]
        
            steps.append({
                'step': 2,
                'description': 'Resolver ecuación característica',
                'equation': f"r = {safe_latex(r)}",
                'method': 'Solución algebraica'
            })
        
            y_func = sp.Function('y')(x)
            solucion_general = Eq(y_func, self.C1 * sp.exp(r * x))
        
            steps.append({
                'step': 3,
                'description': 'Solución general',
                'equation': safe_latex(solucion_general),
                'method': 'Solución para raíz real simple'
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
                        
                            solucion_particular = Eq(y_func, c1_val * sp.exp(r * x))
                            particular_solution = safe_latex(solucion_particular)
                        
                            additional_steps.append({
                                'step': 3,
                                'description': 'Solución particular',
                                'equation': particular_solution,
                                'method': 'Sustitución de constante'
                            })
                            
                            # Generar datos para el gráfico - CORREGIDO
                            print(f"Generando gráfico para solución: {solucion_particular.rhs}")
                            graph_data = self._generar_datos_matplotlib(solucion_particular.rhs)
                            print(f"Datos del gráfico generados: {graph_data is not None}")

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
            print(f"Error en _resolver_orden_1: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'steps': steps,
                'traceback': traceback.format_exc()
            }

    def _resolver_orden_2(self, expr, condiciones):
        x, y = self.x, self.y
        steps = []
        
        try:
            dy_dx2 = next(d for d in expr.atoms(sp.Derivative) 
                     if d.expr == y and self._get_derivative_order(d) == 2)
            
            coef_y_der2 = expr.coeff(dy_dx2)
            expr_normalizado = expr / coef_y_der2 if coef_y_der2 != 1 else expr
            
            a = expr_normalizado.coeff(sp.Derivative(y, x))
            b = expr_normalizado.coeff(y)
            
            r = sp.Symbol('r')
            ecuacion_caracteristica = Eq(r**2 + a*r + b, 0)
            
            steps.append({
                'step': 1,
                'description': 'Ecuación característica',
                'equation': safe_latex(ecuacion_caracteristica),
                'method': 'Reemplazar y por e^(rx)'
            })
            
            raices = sp.solve(ecuacion_caracteristica, r)
            
            steps.append({
                'step': 2,
                'description': 'Resolver ecuación característica',
                'equation': f"Raíces: {', '.join([safe_latex(r) for r in raices])}",
                'method': 'Fórmula cuadrática'
            })
            
            if len(raices) == 2:
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
            else:
                raise ValueError("La ecuación característica no produjo 2 raíces")
            
            steps.append({
                'step': 3,
                'description': 'Solución general',
                'equation': safe_latex(solucion_general),
                'details': f"Tipo: {tipo_raices}",
                'method': 'Solución según tipo de raíces'
            })
            
            particular_solution = None
            additional_steps = []
            graph_data = None
            
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
                            
                            # Generar datos para el gráfico - CORREGIDO
                            print(f"Generando gráfico para solución de orden 2: {solucion_particular.rhs}")
                            graph_data = self._generar_datos_matplotlib(solucion_particular.rhs)
                            print(f"Datos del gráfico generados para orden 2: {graph_data is not None}")
                            
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

    def _generar_datos_matplotlib(self, solucion_expr):
        try:
            if not solucion_expr:
                print("Expresión de solución está vacía")
                return None

            # Convertir la expresión sympy a una función numérica
            f = lambdify(self.x, solucion_expr, modules=['numpy', 'sympy'])
        
            # Determinar un rango de x adecuado dinámicamente
            x_vals = np.linspace(-5, 5, 500)  # Rango inicial
        
            # Evaluar la función con manejo de errores
            y_vals = []
            x_valid = []
        
            for x_val in x_vals:
                try:
                    y_val = complex(f(x_val))
                    if np.isfinite(y_val) and abs(y_val.imag) < 1e-10:  # Solo parte real
                        y_vals.append(float(y_val.real))
                        x_valid.append(float(x_val))
                    elif np.isfinite(y_val.real):  # Si tiene parte imaginaria significativa
                        continue  # O podríamos graficar parte real e imaginaria por separado
                except (TypeError, ValueError, ZeroDivisionError):
                    continue
        
            if len(y_vals) < 10:
                # Intentar con un rango más pequeño
                x_vals = np.linspace(-1, 1, 200)
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
                # Crear la gráfica con Matplotlib
                plt.figure(figsize=(10, 6))
                plt.plot(x_valid, y_vals, 'r-', linewidth=2, label='Solución Particular')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title('Solución de la Ecuación Diferencial')
                plt.grid(True)
                plt.legend()
                
                # Guardar la gráfica en un buffer
                buf = BytesIO()
                plt.savefig(buf, format='png', dpi=100)
                plt.close()
                
                # Codificar la imagen en base64
                image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                buf.close()
                
                return {
                    'image': f"data:image/png;base64,{image_base64}",
                    'x_values': x_valid,
                    'y_values': y_vals
                }
            else:
                print("No se generaron suficientes puntos válidos")
                return None
            
        except Exception as e:
            print(f"Error crítico en _generar_datos_matplotlib: {str(e)}")
            return None
        

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

def resolver_homogeneas(data):
    resolvedor = ResolvedorEcuacionesHomogeneas()
    return resolvedor.resolver_homogeneas(data)