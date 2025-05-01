import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from flask import Flask, request, send_file, render_template
import io
import os

app = Flask(__name__)

# JIT-optimized Mandelbrot function
@jit(nopython=True)
def mandelbrot(c, max_iter):
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z*z + c
        n += 1
    return n

# JIT-optimized Mandelbrot set generator
@jit(nopython=True)
def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    n3 = np.empty((width, height))
    for i in range(width):
        for j in range(height):
            n3[i, j] = mandelbrot(r1[i] + 1j * r2[j], max_iter)
    return n3

# JIT-optimized Julia set generator
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

def plot_both_sets(user_point):
    # Plot settings
    width, height = 800, 800
    max_iter = 800

    # Mandelbrot
    mandel_xmin, mandel_xmax, mandel_ymin, mandel_ymax = -1.5, 0.5, -1, 1
    mandelbrot_data = mandelbrot_set(mandel_xmin, mandel_xmax, mandel_ymin, mandel_ymax, width, height, max_iter)

    # Julia
    julia_xmin, julia_xmax, julia_ymin, julia_ymax = -1.5, 1.5, -1.5, 1.5
    julia_data = julia_set(user_point, julia_xmin, julia_xmax, julia_ymin, julia_ymax, width, height, 256)

    fig, axs = plt.subplots(2, 1, figsize=(10, 20))

    axs[0].imshow(mandelbrot_data.T, extent=[mandel_xmin, mandel_xmax, mandel_ymin, mandel_ymax], cmap='hot')
    axs[0].set_title("Mandelbrot Set")
    axs[0].set_xlabel("Re")
    axs[0].set_ylabel("Im")
    axs[0].plot(user_point.real, user_point.imag, 'ro', markersize=10)
    axs[0].grid(True)

    axs[1].imshow(julia_data.T, extent=[julia_xmin, julia_xmax, julia_ymin, julia_ymax], cmap='hot', interpolation='bilinear')
    axs[1].set_title(f"Julia Set for c = {user_point}")
    axs[1].set_xlabel("Re")
    axs[1].set_ylabel("Im")
    axs[1].grid(True)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/mandelbrot')
def mandelbrot_img():
    try:
        c = complex(request.args.get('c', '0+0j'))
    except ValueError:
        c = 0 + 0j
    return send_file(plot_both_sets(c), mimetype='image/png')

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)

