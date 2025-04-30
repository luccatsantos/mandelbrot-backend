from flask import Flask, request, send_file, render_template
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import io

app = Flask(__name__)

@jit(nopython=True)
def mandelbrot(c, max_iter):
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z*z + c
        n += 1
    return n

@jit(nopython=True)
def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    n3 = np.empty((width, height))
    for i in range(width):
        for j in range(height):
            n3[i, j] = mandelbrot(r1[i] + 1j * r2[j], max_iter)
    return n3

@jit(nopython=True)
def julia_set(c, xmin, xmax, ymin, ymax, width, height, max_iter):
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    n3 = np.empty((width, height))
    for i in range(width):
        for j in range(height):
            z = r1[i] + 1j * r2[j]
            n = 0
            while abs(z) <= 2 and n < max_iter:
                z = z*z + c
                n += 1
            n3[i, j] = n
    return n3

def render_plot(data, extent, point=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(data.T, extent=extent, cmap='hot', interpolation='bilinear')
    if point and extent[0] <= point.real <= extent[1] and extent[2] <= point.imag <= extent[3]:
        ax.plot(point.real, point.imag, 'ro', markersize=5)
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/mandelbrot')
def mandelbrot_img():
    c = complex(request.args.get('c', '0+0j'))
    data = mandelbrot_set(-1.5, 0.5, -1, 1, 600, 600, 500)
    return send_file(render_plot(data, (-1.5, 0.5, -1, 1), point=c), mimetype='image/png')

@app.route('/julia')
def julia_img():
    c = complex(request.args.get('c', '0+0j'))
    data = julia_set(c, -1.5, 1.5, -1.5, 1.5, 600, 600, 300)
    return send_file(render_plot(data, (-1.5, 1.5, -1.5, 1.5)), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)