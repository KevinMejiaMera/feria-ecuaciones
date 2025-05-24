import re
import sympy as sp
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations, 
                                       implicit_multiplication_application)
from sympy.core.function import Derivative

class EquationClassifier:
    def __init__(self, equation_str):
        self.equation_str = equation_str
        self.equation = None
        self.lhs = None
        self.rhs = None
        self.variables = {}
        self.classification = {
            'tipo': 'No determinado',
            'orden': 'No determinado',
            'linealidad': 'No determinado',
            'homogeneidad': 'No determinado'
        }
        
        # Inicializar variables comunes
        self.x = sp.symbols('x')
        self.y = sp.Function('y')(self.x)
        
        try:
            self._parse_equation()
            self._classify()
        except Exception as e:
            raise ValueError(f"Error al procesar la ecuación: {str(e)}")
    
    def _parse_equation(self):
        """Parsear la ecuación en partes izquierda y derecha"""
        # Limpiar la ecuación
        eq_cleaned = self.equation_str.replace(' ', '').replace('^', '**')
        
        # Manejar diferentes formatos de derivadas
        # Primero manejar notación con primas (y', y'', etc.)
        eq_cleaned = re.sub(r"y''''", "Derivative(y, x, 4)", eq_cleaned)  # y''''
        eq_cleaned = re.sub(r"y'''", "Derivative(y, x, 3)", eq_cleaned)   # y'''
        eq_cleaned = re.sub(r"y''", "Derivative(y, x, 2)", eq_cleaned)    # y''
        eq_cleaned = re.sub(r"y'", "Derivative(y, x, 1)", eq_cleaned)     # y'
        
        # Luego manejar notación Leibniz (dy/dx, d2y/dx2, etc.)
        eq_cleaned = re.sub(
            r'd([0-9]*)y/d([0-9]*)x', 
            lambda m: f"Derivative(y, x, {int(m.group(1)) if m.group(1) else 1})", 
            eq_cleaned
        )
        eq_cleaned = re.sub(r'dy/dx', "Derivative(y, x, 1)", eq_cleaned)
        
        # Configurar transformaciones para el parser
        transformations = (standard_transformations + 
                         (implicit_multiplication_application,))
        
        # Definir símbolos locales
        local_dict = {
            'y': self.y,
            'x': self.x,
            'Derivative': Derivative,
            'Eq': sp.Eq,
            'd': sp.Symbol('d'),  # Para evitar conflictos con 'd' en dy/dx
            'sin': sp.sin,
            'cos': sp.cos,
            'tan': sp.tan,
            'exp': sp.exp,
            'log': sp.log
        }
        
        # Separar en LHS y RHS
        if '=' in eq_cleaned:
            lhs_str, rhs_str = eq_cleaned.split('=', 1)
            self.lhs = parse_expr(lhs_str, local_dict=local_dict, transformations=transformations)
            self.rhs = parse_expr(rhs_str, local_dict=local_dict, transformations=transformations)
            self.equation = sp.Eq(self.lhs, self.rhs)
        else:
            # Asumir que es dy/dx = ...
            self.lhs = parse_expr(eq_cleaned, local_dict=local_dict, transformations=transformations)
            self.rhs = sp.Integer(0)
            self.equation = sp.Eq(self.lhs, self.rhs)
        
        # Verificar que la ecuación se parseó correctamente
        if not isinstance(self.equation, sp.Eq):
            raise ValueError("La ecuación no se pudo parsear correctamente. Verifica la sintaxis.")
    
    def _get_order(self):
        """Determinar el orden de la ecuación diferencial"""
        max_order = 0
        for term in sp.preorder_traversal(self.equation):
            if isinstance(term, Derivative):
                order = term.derivative_count
                if order > max_order:
                    max_order = order
        return max_order
    
    def _get_all_y_terms(self):
        """Obtener todos los términos que involucran y y sus derivadas"""
        y_terms = set()
        
        # Agregar y
        y_terms.add(self.y)
        
        # Agregar todas las derivadas que aparecen en la ecuación
        for term in sp.preorder_traversal(self.equation):
            if isinstance(term, Derivative) and term.args[0] == self.y:
                y_terms.add(term)
        
        return y_terms
    
    def _is_linear(self):
        """
        Determinar si la ecuación es lineal según las reglas:
        1. No hay productos entre y y sus derivadas
        2. No hay funciones no lineales de y o sus derivadas
        3. No hay potencias distintas de 1 en y o sus derivadas
        4. Los coeficientes dependen solo de x (o son constantes)
        """
        # Mover todo al lado izquierdo
        expr = self.lhs - self.rhs
        
        # Obtener todos los términos que involucran y y sus derivadas
        y_terms = self._get_all_y_terms()
        
        # 1. Verificar que la expresión sea lineal en y y sus derivadas
        if not self._check_linear_form(expr, y_terms):
            return False
        
        # 2. Verificar coeficientes (deben depender solo de x)
        if not self._check_coefficients_dependency(expr, y_terms):
            return False
        
        return True
    
    def _check_linear_form(self, expr, y_terms):
        """
        Verificar que la expresión sea de la forma:
        a_n(x) * y^(n) + a_(n-1)(x) * y^(n-1) + ... + a_1(x) * y' + a_0(x) * y + f(x) = 0
        """
        try:
            # Expandir la expresión
            expanded = sp.expand(expr)
            
            # Verificar cada término en la suma
            for term in sp.Add.make_args(expanded):
                if self._contains_y_or_derivatives(term, y_terms):
                    if not self._is_term_linear(term, y_terms):
                        return False
            
            return True
        except:
            return False
    
    def _contains_y_or_derivatives(self, term, y_terms):
        """Verificar si un término contiene y o sus derivadas"""
        for y_term in y_terms:
            if term.has(y_term):
                return True
        return False
    
    def _is_term_linear(self, term, y_terms):
        """
        Verificar si un término individual es lineal:
        - Debe ser de la forma: coef(x) * y_term^1
        - No puede tener productos de diferentes términos y
        - No puede tener funciones no lineales
        - No puede tener potencias != 1
        """
        # Contar cuántos términos y diferentes aparecen en este término
        y_terms_in_term = []
        for y_term in y_terms:
            if term.has(y_term):
                y_terms_in_term.append(y_term)
        
        # Si no hay términos y, es solo función de x (válido)
        if len(y_terms_in_term) == 0:
            return True
        
        # Si hay más de un tipo de término y, verificar que no sea un producto
        if len(y_terms_in_term) > 1:
            # Verificar si es un producto de diferentes términos y
            if self._is_product_of_y_terms(term, y_terms_in_term):
                return False
        
        # Verificar cada término y que aparece
        for y_term in y_terms_in_term:
            # Verificar potencias
            if not self._check_power_of_y_term(term, y_term):
                return False
            
            # Verificar funciones no lineales
            if not self._check_no_nonlinear_functions(term, y_term):
                return False
        
        return True
    
    def _is_product_of_y_terms(self, term, y_terms_in_term):
        """Verificar si el término es un producto de diferentes términos y"""
        # Intentar factorizar para ver si hay productos
        try:
            factors = sp.Mul.make_args(term)
            y_factor_count = 0
            
            for factor in factors:
                for y_term in y_terms_in_term:
                    if factor.has(y_term) and factor != y_term:
                        # Si el factor contiene y_term pero no es exactamente y_term,
                        # podría ser una potencia o función
                        continue
                    elif factor == y_term:
                        y_factor_count += 1
            
            # Si hay más de un factor que es exactamente un término y, es un producto
            return y_factor_count > 1
        except:
            # En caso de error, asumir que podría ser un producto (no lineal)
            return True
    
    def _check_power_of_y_term(self, term, y_term):
        """Verificar que y_term aparece solo con potencia 1"""
        try:
            # Verificar si hay potencias explícitas
            for subterm in sp.preorder_traversal(term):
                if isinstance(subterm, sp.Pow):
                    base, exp = subterm.args
                    if base == y_term and exp != 1:
                        return False
                    # También verificar potencias de expresiones que contienen y_term
                    if base.has(y_term) and exp != 1:
                        return False
            
            return True
        except:
            return False
    
    def _check_no_nonlinear_functions(self, term, y_term):
        """Verificar que no hay funciones no lineales de y_term"""
        try:
            nonlinear_functions = [sp.sin, sp.cos, sp.tan, sp.exp, sp.log, 
                                 sp.sinh, sp.cosh, sp.tanh, sp.asin, sp.acos, sp.atan]
            
            for subterm in sp.preorder_traversal(term):
                for func in nonlinear_functions:
                    if isinstance(subterm, func) and subterm.args[0].has(y_term):
                        return False
            
            return True
        except:
            return False
    
    def _check_coefficients_dependency(self, expr, y_terms):
        """Verificar que los coeficientes solo dependen de x"""
        try:
            expanded = sp.expand(expr)
            
            for y_term in y_terms:
                # Obtener el coeficiente de cada término y
                coeff = expanded.coeff(y_term)
                if coeff is not None and coeff != 0:
                    # El coeficiente no debe contener y ni sus derivadas
                    for other_y_term in y_terms:
                        if coeff.has(other_y_term):
                            return False
            
            return True
        except:
            return False
    
    def _is_homogeneous(self):
        """Determinar si la ecuación es homogénea"""
        # Para ecuaciones lineales
        if self.classification['linealidad'] == 'Lineal':
            return self.rhs == 0
        else:
            # Para ecuaciones no lineales, verificar si todos los términos son del mismo grado
            try:
                k = sp.symbols('k')
                substituted = self.equation.subs({self.y: k*self.y, self.x: k*self.x})
                simplified = sp.simplify(substituted)
                
                if simplified.lhs == (k**sp.Wild('n')) * self.equation.lhs and \
                   simplified.rhs == (k**sp.Wild('n')) * self.equation.rhs:
                    return True
                else:
                    return False
            except:
                return False
    
    def _classify(self):
        """Realizar la clasificación completa"""
        # Determinar tipo (ordinaria o parcial)
        if 'd^2' in self.equation_str or 'partial' in self.equation_str.lower():
            self.classification['tipo'] = 'Parcial'
        else:
            self.classification['tipo'] = 'Ordinaria'
        
        # Determinar orden
        order = self._get_order()
        self.classification['orden'] = str(order)
        
        # Determinar linealidad
        self.classification['linealidad'] = 'Lineal' if self._is_linear() else 'No lineal'
        
        # Determinar homogeneidad
        self.classification['homogeneidad'] = 'Homogénea' if self._is_homogeneous() else 'No homogénea'
    
    def get_classification(self):
        """Obtener el diccionario de clasificación"""
        return self.classification


# Función para integrar con Flask o otros frameworks
def classify_differential_equation(equation_str):
    try:
        classifier = EquationClassifier(equation_str)
        return {
            'status': 'success',
            'classification': classifier.get_classification()
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }


# Función de prueba para validar el clasificador
def test_classifier():
    """Función para probar el clasificador con varios ejemplos"""
    test_cases = [
        # Casos lineales
        ("y'' + 2*y' + y = 0", "Lineal"),  # Homogénea lineal
        ("y'' + x*y' + y = x", "Lineal"),  # No homogénea lineal
        ("2*y' + 3*y = sin(x)", "Lineal"), # Primer orden lineal
        
        # Casos no lineales
        ("y*y' = x", "No lineal"),         # Producto y*y'
        ("y'' + (y')^2 = 0", "No lineal"), # Potencia de derivada
        ("y'' + y^2 = 0", "No lineal"),    # Potencia de y
        ("y' + sin(y) = 0", "No lineal"),  # Función no lineal de y
        ("y*y'' = 1", "No lineal"),        # Producto y*y''
        ("y' + y*cos(y) = 0", "No lineal") # Coeficiente depende de y
    ]
    
    print("Probando clasificador de ecuaciones diferenciales:")
    print("=" * 60)
    
    for eq_str, expected in test_cases:
        try:
            result = classify_differential_equation(eq_str)
            if result['status'] == 'success':
                actual = result['classification']['linealidad']
                status = "✓" if actual == expected else "✗"
                print(f"{status} {eq_str:<25} | Esperado: {expected:<10} | Obtenido: {actual}")
            else:
                print(f"✗ {eq_str:<25} | Error: {result['message']}")
        except Exception as e:
            print(f"✗ {eq_str:<25} | Excepción: {str(e)}")

#test_classifier()