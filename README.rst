.. |pypi| image:: https://img.shields.io/pypi/v/pysmithchart?color=68CA66
   :target: https://pypi.org/project/pysmithchart/
   :alt: PyPI

.. |github| image:: https://img.shields.io/github/v/tag/scottprahl/pysmithchart?label=github&color=68CA66
   :target: https://github.com/scottprahl/pysmithchart
   :alt: GitHub

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/pysmithchart?label=conda&color=68CA66
   :target: https://github.com/conda-forge/pysmithchart-feedstock
   :alt: Conda

.. |doi| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.18409151.svg
   :target: https://doi.org/10.5281/zenodo.18409151
   :alt: doi  

.. |license| image:: https://img.shields.io/github/license/scottprahl/pysmithchart?color=68CA66
   :target: https://github.com/scottprahl/pysmithchart/blob/main/LICENSE.txt
   :alt: License

.. |test| image:: https://github.com/scottprahl/pysmithchart/actions/workflows/test.yaml/badge.svg
   :target: https://github.com/scottprahl/pysmithchart/actions/workflows/test.yaml
   :alt: Testing

.. |docs| image:: https://readthedocs.org/projects/pysmithchart/badge?color=68CA66
   :target: https://pysmithchart.readthedocs.io
   :alt: Documentation

.. |downloads| image:: https://img.shields.io/pypi/dm/pysmithchart?color=68CA66
   :target: https://pypi.org/project/pysmithchart/
   :alt: Downloads

.. |lite| image:: https://img.shields.io/badge/try-JupyterLite-68CA66.svg
   :target: https://scottprahl.github.io/pysmithchart/
   :alt: Try Online

pysmithchart
============

|pypi| |github| |conda| |doi|

|license| |test| |docs| |downloads|

|lite|

**pysmithchart** is a Python library that provides high-quality Smith charts for RF and microwave engineering applications. Built as a native extension to **matplotlib**, it enables reproducible analysis and publication-ready visualization of reflection coefficients, impedances, and admittances commonly encountered in transmission-line theory, antenna characterization, and network analysis.

Smith charts remain a foundational tool in RF engineering, and this library is designed to support both instructional use and research workflows. By integrating directly with matplotlib's projection system, pysmithchart enables familiar plotting syntax while offering fine-grained control of chart geometry, grid styling, interpolation, and layout.

.. image:: https://raw.githubusercontent.com/scottprahl/pysmithchart/main/docs/images/readme_fig3.svg
   :alt: Smith Chart Example
   :width: 400px
   :align: center

This project originated from `pySmithPlot <https://github.com/vMeijin/pySmithPlot>`_ by Paul Staerke.  
It has been completely rewritten so any issues are not his.

----

Features
--------

* **Seamless matplotlib integration** — implemented as a projection; compatible with standard plotting workflows
* **Support for common RF quantities** — reflection coefficients, impedances, admittances, and S-parameters
* **Configurable analytical grids** — control spacing, style, and precision of constant-R and constant-X curves
* **Interpolation utilities** — optional smoothing and resampling of complex-valued datasets
* **Custom marker rotation** — useful for frequency-indexed trajectories and multi-point measurement data
* **Publication-quality output** — full control over fonts, colors, annotations, and line styling

Installation
------------

**Using pip:**

.. code-block:: bash

    pip install pysmithchart

**Using conda:**

.. code-block:: bash

    conda install -c conda-forge pysmithchart

Quick Start
-----------

**Reflection Coefficients (S-Parameters)**

.. code-block:: python

    import matplotlib.pyplot as plt
    from pysmithchart import R_DOMAIN

    S = [0.5 + 0.3j, -0.2 - 0.1j]

    plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1, projection="smith")
    plt.plot(S, domain=R_DOMAIN, marker='o', markersize=10, label='S₁₁')
    plt.legend()
    plt.title('Reflection Coefficients')
    plt.show()

.. image:: https://raw.githubusercontent.com/scottprahl/pysmithchart/main/docs/images/readme_fig1.svg
   :alt: simple smithchart
   :width: 400px
   :align: center

**Normalized Impedance Example**

.. code-block:: python

    import matplotlib.pyplot as plt
    import pysmithchart

    ZL = [30 + 30j, 50 + 50j, 100 + 100j]

    sc = {"grid.Z.minor_enable":True}
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(1, 1, 1, projection="smith", Z0=200, **sc)
    plt.plot(ZL, "b-o", markersize=10, label='Load Impedance')
    plt.legend()
    plt.title('Impedances with Z₀ = 200Ω')
    plt.show()

.. image:: https://raw.githubusercontent.com/scottprahl/pysmithchart/main/docs/images/readme_fig2.svg
   :alt: simple impedance example
   :width: 400px
   :align: center

Documentation
-------------

Comprehensive documentation, including the API reference, tutorials, theoretical background, and worked examples, is available at:

    https://pysmithchart.readthedocs.io

A live, browser-based environment powered by JupyterLite is available for experimentation without installation:

    https://scottprahl.github.io/pysmithchart/

License
-------

``pysmithchart`` is released under the BSD-3 Clause License.

Citation
--------

If you use ``pysmithchart`` in academic or technical work, please cite:

Prahl, S. (2026). *pysmithchart: A Python module for Smith charts* (Version 0.9.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.18409151

BibTeX
~~~~~~

.. code-block:: bibtex

   @software{pygrin_prahl_2026,
     author    = {Scott Prahl and Paul Staerke},
     title     = {pysmithchart: A Python module for Smith charts},
     year      = {2026},
     version   = {0.9.0},
     doi       = {10.5281/zenodo.18409151},
     url       = {https://github.com/scottprahl/pysmithchart},
     publisher = {Zenodo}
   }
