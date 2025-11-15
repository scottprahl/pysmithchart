"""
JupyterLite configuration for Pyodide-based execution (JupyterLite ≥ 0.6.0)

This file replaces earlier configuration patterns used in JupyterLite 0.4.x and 0.5.x,
where multiple files such as `jupyter-lite.json`, `lab/jupyter-lite.json`, and various
hand-edited runtime manifests were required.

Starting with JupyterLite 0.6.0, **this is the only configuration file that should be
authored and versioned**. All runtime configuration files (including `jupyter-lite.json`,
kernel metadata, and the `lab/` configuration tree) are now automatically generated
during the build step:

    jupyter lite build

The generated output is written into the directory specified by `LiteBuildConfig`
(typically `lite/`) and may contain:

    - jupyter-lite.json
    - lab/jupyter-lite.json
    - /api/kernelspecs/...
    - bundled wheels or assets

These generated files should **not be edited manually** and should generally **not be
committed to version control**, unless the generated site itself is being published
(e.g., to GitHub Pages).

In short:

    ✔ Keep and edit this file (`jupyter_lite_config.py`)
    ❌ Remove and do not maintain old config files such as:
         - jupyter-lite.json
         - lab/jupyter-lite.json

This single config file defines both the runtime environment (JupyterLab UI, base URL,
kernel assignment) and the build settings (contents, output directory, extensions) in
a unified way compatible with modern JupyterLite releases.
"""

c.LiteBuildConfig = {
    "apps": ["lab"],
    "contents": ["docs"],
    "output_dir": "lite",
}

# Runtime settings (replace your hard-coded baseUrl with auto-detect "./")
c.JupyterLiteConfig = {
    "jupyter-config-data": {
        "appName": "PySmithChart",
        "appUrl": "./lab",
        "baseUrl": "./",   # works local + GitHub Pages
        "defaultKernelName": "python",
    }
}

# Optional: include dev lab extensions from venv
import os
venv_base = os.path.join(os.getcwd(), ".venv")
labext_path = os.path.join(venv_base, "share/jupyter/labextensions")
c.FederatedExtensionAddon.extra_labextensions_path = [labext_path]
