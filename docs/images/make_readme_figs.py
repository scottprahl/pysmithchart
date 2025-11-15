#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from pysmithchart import S_PARAMETER

S = [0.5 + 0.3j, -0.2 - 0.1j]

plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1, projection="smith", grid_major_color_y="blue")
plt.plot(S, ls="", markersize=10, datatype=S_PARAMETER)
plt.title("Plotting Complex Reflection Coefficients")
plt.savefig("readme_fig1.svg", format="svg")
plt.show()


ZL = [30 + 30j, 50 + 50j, 100 + 100j]

plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1, projection="smith", axes_impedance=200, grid_minor_enable=True)
plt.plot(ZL, "b-o", markersize=10)  # default datatype is Z_PARAMETER
plt.title("Impedance parameters with Z₀=200Ω")
plt.savefig("readme_fig2.svg", format="svg")
plt.show()


ZL = [40 + 20j, 60 + 80j, 90 + 30j]

plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1, projection="smith")
plt.plot(ZL, markersize=16, ls="--", markerhack=True, rotate_marker=True)
plt.title("Custom markers")
plt.savefig("readme_fig3.svg", format="svg")
plt.show()
