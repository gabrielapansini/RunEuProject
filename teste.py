from flask import Flask, render_template, request
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sympy import symbols, lambdify, cos, sin
from matplotlib.figure import Figure
import numpy as np
import io
import matplotlib.pyplot as plt
from io import BytesIO
import base64


app = Flask(__name__)

##newton method for root equations
def newton_method(f, df, x0, tolerance=1e-6, max_iterations=100):
    iterations = []
    x = x0
    for i in range(max_iterations):
        x_new = x - f(x) / df(x)
        iterations.append((i + 1, x, abs(x_new - x), f(x)))
        if abs(x_new - x) < tolerance:
            return x_new, iterations
        x = x_new
    return None, iterations

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ballgame")
def ballgame():
    return render_template("ballgame.html")



@app.route("/root-finding", methods=["GET", "POST"])
def root_finder():
    if request.method == 'POST':
        try:
            x0 = float(request.form['x0'])
            expression = request.form['expression']
            derivative = request.form['derivative']

            x = symbols('x')
            f_expr = eval(expression)
            df_expr = eval(derivative)
            f = lambdify(x, f_expr)
            df = lambdify(x, df_expr)

            root, iterations = newton_method(f, df, x0)

            if root is not None:
                # Render the template with the result data
                return render_template('root_finding.html', result=root, iterations=iterations)
            else:
                # Render the template with an error message
                return render_template('root_finding.html', error="Root not found within the maximum number of iterations.")
        except Exception as e:
            # Render the template with an error message for input validation
            return render_template('root_finding.html', error="An error occurred. Please check your input.")

   
    return render_template("root_finding.html")




#interpolation

def linear_interpolation(x_values, y_values, interpolation_point):
    n = len(x_values)
    for i in range(n - 1):
        if x_values[i] <= interpolation_point <= x_values[i + 1]:
            slope = (y_values[i + 1] - y_values[i]) / (x_values[i + 1] - x_values[i])
            interpolated_value = y_values[i] + slope * (interpolation_point - x_values[i])
            break
    else:
        return None, None, None  # Return None values if the interpolation point is outside the range

    # Generate graph points for linear interpolation
    graph_x = np.linspace(x_values.min(), x_values.max(), 100)
    graph_y = []
    for x in graph_x:
        for i in range(n - 1):
            if x_values[i] <= x <= x_values[i + 1]:
                slope = (y_values[i + 1] - y_values[i]) / (x_values[i + 1] - x_values[i])
                graph_y.append(y_values[i] + slope * (x - x_values[i]))
                break

    return interpolated_value, graph_x, graph_y

##polynomial 

def polynomial_interpolation(x_values, y_values, interpolation_point):
    n = len(x_values)
    coefficients = np.zeros(n)

    for i in range(n):
        coefficients[i] = y_values[i]

    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coefficients[i] = (coefficients[i] - coefficients[i - 1]) / (x_values[i] - x_values[i - j])

    result = 0.0
    for i in range(n - 1, -1, -1):
        result = coefficients[i] + (interpolation_point - x_values[i]) * result

    # Generate graph points for polynomial interpolation
    graph_x = np.linspace(x_values.min(), x_values.max(), 100)
    graph_y = np.zeros_like(graph_x)
    for i in range(n):
        graph_y += coefficients[i] * np.power(graph_x, n - i - 1)

    return result, graph_x, graph_y



@app.route("/interpolation-finding", methods=["POST", "GET"])
def interpolation_finder():
    if request.method == 'POST':
        x_values_str = request.form['x_values']
        y_values_str = request.form['y_values']
        interpolation_point = float(request.form['interpolation_point'])

        x_values = np.array([float(x) for x in x_values_str.split(',')])
        y_values = np.array([float(y) for y in y_values_str.split(',')])

        # Linear interpolation
        linear_result, linear_graph_x, linear_graph_y = linear_interpolation(x_values, y_values, interpolation_point)

        # Polynomial interpolation
        polynomial_result, polynomial_graph_x, polynomial_graph_y = polynomial_interpolation(x_values, y_values, interpolation_point)


        # Handle the case when interpolation point is outside the range for linear interpolation
        if linear_result is None:
            linear_graph_data = None
        else:
            # Generating the linear interpolation graph
            plt.plot(x_values, y_values, 'o', label='Data Points')
            plt.plot(linear_graph_x, linear_graph_y, label='Linear Interpolation')
            plt.plot(interpolation_point, linear_result, 'ro', label='Interpolated Value (Linear)')
            plt.xlabel('X Values')
            plt.ylabel('Y Values')
            plt.title('Linear Interpolation')
            plt.legend()
            plt.grid(True)

            buffer_linear = BytesIO()
            plt.savefig(buffer_linear, format='png')
            buffer_linear.seek(0)
            linear_graph_data = base64.b64encode(buffer_linear.getvalue()).decode()
            plt.close()

        # Handle the case when interpolation point is outside the range for polynomial interpolation
        if polynomial_result is None:
            polynomial_graph_data = None
        else:
            # Generating the polynomial interpolation graph
            plt.plot(x_values, y_values, 'o', label='Data Points')
            plt.plot(polynomial_graph_x, polynomial_graph_y, label='Polynomial Interpolation')
            plt.plot(interpolation_point, polynomial_result, 'ro', label='Interpolated Value (Polynomial)')
            plt.xlabel('X Values')
            plt.ylabel('Y Values')
            plt.title('Polynomial Interpolation')
            plt.legend()
            plt.grid(True)

            buffer_polynomial = BytesIO()
            plt.savefig(buffer_polynomial, format='png')
            buffer_polynomial.seek(0)
            polynomial_graph_data = base64.b64encode(buffer_polynomial.getvalue()).decode()
            plt.close()

        return render_template("interpolation_finding.html", 
                               linear_result=linear_result, 
                               linear_graph_data=linear_graph_data,
                               polynomial_result=polynomial_result, 
                               polynomial_graph_data=polynomial_graph_data)

    return render_template("interpolation_finding.html")



##ballGame

if __name__ == "__main__":
    app.run(debug=True)

