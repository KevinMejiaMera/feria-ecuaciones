import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import matplotlib
matplotlib.use('Agg')  # Configurar matplotlib para no usar GUI
from sympy import symbols, Function, Eq, parse_expr, diff
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application
import traceback
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import os
# Configuración inicial
sys.path.append(str(Path(__file__).parent))
app = Flask(__name__)
app.secret_key = 'tu_clave_secreta_aqui'

# Importaciones de controladores
from controllers.ecuaciones.lineales import resolver_lineales
from controllers.ecuaciones import homogeneas, no_homogeneas 
from controllers.derivadas_controller import calcular_derivada
from controllers.integrales_controller import calcular_integral
from utils.math_utils import safe_latex
from controllers.ecuaciones.clasificacion import classify_differential_equation

# ==============================================
# RUTAS PRINCIPALES
# ==============================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/derivadas')
def derivadas():
    return render_template('derivadas.html')

@app.route('/integrales')
def integrales():
    return render_template('integrales.html')

# ==============================================
# RUTAS PARA CÁLCULOS
# ==============================================

@app.route('/calcular-derivada', methods=['POST'])
def calcular_derivada_route():
    data = request.get_json()
    return calcular_derivada(data)

@app.route('/calcular-integral', methods=['POST'])
def calcular_integral_route():
    data = request.get_json()
    return calcular_integral(data)

# ==============================================
# RUTAS PARA ECUACIONES DIFERENCIALES
# ==============================================

@app.route('/ecuaciones')
def ecuaciones():
    return render_template('ecuaciones/menu.html')

@app.route('/clasificar-ecuacion', methods=['GET', 'POST'])
def clasificar_ecuacion():
    if request.method == 'GET':
        return render_template('ecuaciones/clasificacion.html')
    else:
        if request.is_json:
            data = request.get_json()
            equation = data.get('equation', '').strip()
        else:
            equation = request.form.get('equation', '').strip()
        
        if not equation:
            return jsonify({
                'status': 'error',
                'message': 'No se proporcionó una ecuación'
            }), 400
        
        try:
            result = classify_differential_equation(equation)
            return jsonify(result)
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Error al clasificar la ecuación: {str(e)}'
            }), 500

@app.route('/ecuaciones/homogeneas', methods=['GET'])
def ecuaciones_homogeneas():
    """Ruta para mostrar la interfaz de ecuaciones homogéneas"""
    return render_template('ecuaciones/resolver_homogeneas.html')

@app.route('/ecuaciones/homogeneas', methods=['POST'])
def api_ecuaciones_homogeneas():
    try:
        data = request.get_json()
        if not data or 'equation' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Datos de entrada inválidos'
            }), 400

        resolvedor = homogeneas.ResolvedorEcuacionesHomogeneas()
        result = resolvedor.resolver_homogeneas(data)
        
        # Limpiar cualquier figura de matplotlib que haya quedado en memoria
        plt.close('all')
        
        return jsonify(result)

    except Exception as e:
        print(f"Error en api_ecuaciones_homogeneas: {str(e)}")
        traceback.print_exc()
        # Asegurarse de cerrar cualquier figura que pueda haber quedado abierta
        plt.close('all')
        return jsonify({
            'status': 'error',
            'message': f'Error interno del servidor: {str(e)}'
        }), 500


@app.route('/ecuaciones/no-homogeneas', methods=['GET'])
def ecuaciones_no_homogeneas():
    """Ruta para mostrar la interfaz de ecuaciones no homogéneas"""
    return render_template('ecuaciones/resolver_no_homogeneas.html')

@app.route('/ecuaciones/no-homogeneas', methods=['POST'])
def api_ecuaciones_no_homogeneas():
    try:
        data = request.get_json()
        if not data or 'equation' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Datos de entrada inválidos',
                'sugerencia': 'Por favor ingrese una ecuación diferencial válida'
            }), 400

        # Validación adicional del input
        equation = data['equation'].strip()
        if not equation or len(equation) < 3:
            return jsonify({
                'status': 'error',
                'message': 'La ecuación es demasiado corta o vacía',
                'sugerencia': 'Ejemplos válidos: "y\' + y = e^x" o "y\'\' + 4y = sin(x)"'
            }), 400

        # Verificar que tenga el formato básico de ecuación no homogénea
        if '=' not in equation:
            return jsonify({
                'status': 'error',
                'message': 'La ecuación debe contener un signo igual (=)',
                'sugerencia': 'Formato correcto: "lado izquierdo = lado derecho"'
            }), 400

        # Verificar que el lado derecho no sea cero
        lhs, rhs = equation.split('=', 1)
        if rhs.strip() == '0':
            return jsonify({
                'status': 'error',
                'message': 'El lado derecho es cero (ecuación homogénea)',
                'sugerencia': 'Use el resolvedor de ecuaciones homogéneas para este caso'
            }), 400

        result = no_homogeneas.resolver_no_homogeneas(data)
        
        # Mejorar mensajes de error para el usuario
        if result.get('status') == 'error':
            error_msg = result.get('message', 'Error desconocido')
            suggestion = result.get('sugerencia', 'Revise la ecuación e intente nuevamente')
            
            # Casos especiales con sugerencias específicas
            if "coeficientes constantes" in error_msg:
                suggestion = "Solo se admiten ecuaciones con coeficientes constantes (números)"
            elif "parte homogénea" in error_msg:
                suggestion = "La parte homogénea de la ecuación no pudo resolverse. Verifique los coeficientes"
            
            return jsonify({
                'status': 'error',
                'message': error_msg,
                'sugerencia': suggestion,
                'steps': result.get('steps', [])
            }), 400
        
        return jsonify(result)

    except Exception as e:
        print(f"Error en api_ecuaciones_no_homogeneas: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Error interno al procesar la ecuación: {str(e)}',
            'sugerencia': 'Revise la sintaxis de su ecuación e intente nuevamente'
        }), 500
# ==============================================
# FUNCIONES AUXILIARES
# ==============================================

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))