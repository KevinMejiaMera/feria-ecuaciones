import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sympy import lambdify, symbols, sympify, Float, Expr, S, simplify
import logging
from sympy.utilities.lambdify import lambdastr
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_plot(eq, solution, initial_conditions=None):
    """Genera un gráfico de la solución de la ecuación diferencial"""
    try:
        x = symbols('x')
        y = symbols('y', cls=Function)
        
        # Obtener la expresión de la solución
        if isinstance(solution, Expr):
            sol_expr = solution
        else:
            sol_expr = solution.rhs if hasattr(solution, 'rhs') else solution
        
        # Intentar simplificar la expresión
        try:
            sol_expr = simplify(sol_expr)
        except Exception as e:
            logger.warning(f"No se pudo simplificar la expresión: {str(e)}")
        
        # Verificar si hay constantes arbitrarias en la solución
        if str(sol_expr).find('C1') != -1 or str(sol_expr).find('C2') != -1:
            if not initial_conditions:
                logger.warning("La solución contiene constantes arbitrarias sin condiciones iniciales")
                return None
        
        # Obtener una representación de la expresión como string
        expr_str = str(sol_expr)
        logger.info(f"Expresión a graficar: {expr_str}")
        
        # Detectar funciones especiales no soportadas directamente por NumPy
        unsupported_funcs = ['LambertW', 'besselj', 'bessely', 'airy', 'elliptic']
        for func in unsupported_funcs:
            if func in expr_str:
                logger.warning(f"Función no soportada detectada: {func}")
                return None
        
        # Convertir a función numérica con manejo de errores mejorado
        try:
            # Crear una función lambda más robusta con módulos adicionales
            modules = [
                'numpy',
                {
                    'exp': np.exp,
                    'sin': np.sin,
                    'cos': np.cos,
                    'tan': np.tan,
                    'log': np.log,
                    'sqrt': np.sqrt,
                    'atan': np.arctan,
                    'asin': np.arcsin,
                    'acos': np.arccos,
                    'Abs': np.abs,
                    'ln': np.log,
                    'pi': np.pi,
                    'E': np.e,
                    'atan2': np.arctan2,
                    'sinh': np.sinh,
                    'cosh': np.cosh,
                    'tanh': np.tanh,
                    'asinh': np.arcsinh,
                    'acosh': np.arccosh,
                    'atanh': np.arctanh,
                    'ceiling': np.ceil,
                    'floor': np.floor
                }
            ]
            
            f = lambdify(x, sol_expr, modules=modules)
            
            # Verificar si la función es válida con un test simple
            test_value = f(1.0)
            if isinstance(test_value, complex) or np.isnan(test_value) or np.isinf(test_value):
                logger.warning(f"La función genera valores problemáticos en prueba inicial: {test_value}")
        except Exception as e:
            logger.error(f"Error al crear función lambdify: {str(e)}")
            # Intentar otro enfoque
            try:
                # Construir una función segura que capture excepciones
                lambda_str = lambdastr(x, sol_expr)
                # Reemplazar funciones no soportadas
                lambda_str = lambda_str.replace('math.', 'np.')
                
                # Definir una función segura que capture excepciones
                def safe_f(x_val):
                    try:
                        result = eval(lambda_str.replace('x', f'({x_val})'))
                        if isinstance(result, complex):
                            return np.nan if np.isnan(result.real) else result.real
                        return result
                    except Exception:
                        return np.nan
                
                f = safe_f
            except Exception as e2:
                logger.error(f"Error en el método alternativo: {str(e2)}")
                return None
        
        # Generar múltiples conjuntos de datos para encontrar uno viable
        best_x_vals, best_y_vals = None, None
        best_valid_count = 0
        
        # Intentar varios rangos para encontrar el más adecuado
        x_ranges = [
            (-5, 5, 500),       # Rango estándar
            (-10, 10, 600),     # Rango amplio
            (-2, 2, 400),       # Rango estrecho
            (-20, 20, 800),     # Rango muy amplio
            (-0.5, 0.5, 300)    # Rango muy estrecho
        ]
        
        for x_min, x_max, points in x_ranges:
            try:
                # Generar puntos x con pequeña perturbación para evitar singularidades exactas
                x_vals = np.linspace(x_min, x_max, points)
                # Añadir pequeña perturbación aleatoria para evitar valores exactos problemáticos
                x_vals = x_vals + np.random.normal(0, 1e-10, points)
                
                # Calcular valores y con manejo de excepciones para cada punto
                y_vals = np.array([f(float(xi)) for xi in x_vals])
                
                # Convertir complejos a reales si es necesario
                if np.iscomplexobj(y_vals):
                    y_vals = np.real(y_vals)
                
                # Filtrar valores no finitos y excesivamente grandes
                mask = np.isfinite(y_vals) & (np.abs(y_vals) < 1e6)
                valid_count = np.sum(mask)
                
                if valid_count > best_valid_count:
                    best_valid_count = valid_count
                    best_x_vals = x_vals[mask]
                    best_y_vals = y_vals[mask]
                    
                    # Si tenemos suficientes puntos, podemos parar
                    if valid_count > points * 0.7:  # Si más del 70% son válidos
                        break
                        
            except Exception as e:
                logger.warning(f"Error en el rango {x_min} a {x_max}: {str(e)}")
                continue
        
        # Verificamos si tenemos suficientes puntos para graficar
        if best_x_vals is None or len(best_x_vals) < 20:
            logger.error("No se pudieron generar suficientes puntos válidos para graficar")
            return None
            
        # Configuración del gráfico con estilo mejorado
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        plt.style.use('seaborn-v0_8-whitegrid')  # Versión actualizada de seaborn
        ax.set_facecolor('#f8f9fa')
        
        # Graficar solución con estilo profesional
        plt.plot(best_x_vals, best_y_vals, color='#4a6baf', linewidth=2.5, label='Solución')
        
        # Graficar condiciones iniciales si existen
        if initial_conditions:
            for i, ic in enumerate(initial_conditions):
                for point, value in ic.items():
                    try:
                        # Extraer valor x del punto (condición inicial)
                        if hasattr(point, 'args'):
                            x_point = float(point.args[1])
                        else:
                            x_point = float(point)
                            
                        y_value = float(value)
                        
                        # Verificar si el punto está dentro del rango graficado
                        if min(best_x_vals) <= x_point <= max(best_x_vals):
                            plt.scatter(x_point, y_value, color='#e74c3c', s=100, 
                                       zorder=5, label=f'y({x_point}) = {y_value:.2f}')
                            
                            # Añadir línea vertical en el punto inicial
                            plt.axvline(x=x_point, color='#e74c3c', linestyle='--', alpha=0.5)
                    except Exception as e:
                        logger.warning(f"Error al graficar condición inicial: {str(e)}")
                        continue
        
        # Personalización avanzada del gráfico
        plt.title('Solución de la Ecuación Diferencial', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('x', fontsize=12, labelpad=10)
        plt.ylabel('y(x)', fontsize=12, labelpad=10)
        
        # Ajustar el rango y para evitar gráficos muy estirados
        y_range = best_y_vals.max() - best_y_vals.min()
        if y_range < 1e-6:  # Si el rango es muy pequeño
            mean_y = np.mean(best_y_vals)
            plt.ylim(mean_y - 1, mean_y + 1)  # Establecer un rango fijo
        else:
            # Calcular límites con margen
            y_min, y_max = best_y_vals.min(), best_y_vals.max()
            margin = y_range * 0.1  # 10% de margen
            plt.ylim(y_min - margin, y_max + margin)
        
        # Mejorar la leyenda y la cuadrícula
        plt.legend(loc='best', frameon=True, shadow=True)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Añadir bordes al gráfico
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#cccccc')
        
        # Guardar en buffer con alta calidad
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', 
                   facecolor=ax.get_facecolor(), edgecolor='none')
        plt.close()
        buf.seek(0)
        
        # Convertir a base64 para mostrar en HTML
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f'<img src="data:image/png;base64,{img_base64}" class="img-fluid" alt="Gráfico de la solución">'
    
    except Exception as e:
        logger.error(f"Error general generando gráfico: {str(e)}")
        return None
        
# Importar Function si no está definido en el alcance global
try:
    from sympy import Function
except ImportError:
    logger.warning("No se pudo importar Function desde sympy")