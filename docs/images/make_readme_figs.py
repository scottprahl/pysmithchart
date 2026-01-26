#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from pysmithchart import NORM_Z_DOMAIN

S = [0.5 + 0.3j, -0.2 - 0.1j]

sc = {"grid.Z.major.color": "blue"}
plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1, projection="smith", **sc)
plt.plot(S, ls="", markersize=10, domain=NORM_Z_DOMAIN)
plt.title("Plotting Complex Reflection Coefficients")
plt.savefig("readme_fig1.svg", format="svg")
plt.show()


ZL = [30 + 30j, 50 + 50j, 100 + 100j]

sc = {"grid.Z.minor.enable": True}
plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1, projection="smith", Z0=200, **sc)
plt.plot(ZL, "b-o", markersize=10)
plt.title("Impedance parameters with Z₀=200Ω")
plt.savefig("readme_fig2.svg", format="svg")
plt.show()


ZL = [40 + 20j, 60 + 80j, 90 + 30j]

plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1, projection="smith")
plt.plot(ZL, arrow=True)
plt.title("Arrows")
plt.savefig("readme_fig3.svg", format="svg")
plt.show()
