Genetic Algorithm for Lattice Isometry to Z^n
Overview

This project explores the use of a Genetic Algorithm (GA) to solve the Lattice Isometry Problem. Specifically, it aims to determine if a given unimodular integer lattice, represented by its basis matrix B, is isometric to the standard integer lattice Z^n.

The core of the method is a search for a unimodular matrix U in GL(n, Z) that minimizes the objective function:
d = min || G - U^T * U ||_F
where G is the Gram matrix of the input lattice (G = B^T * B).


Setup for IDEs (Spyder, VSCode, PyCharm)

The recommended way to set up this project is to install it in "editable" mode. This makes the lattice_ga library available everywhere in your project, which is ideal for IDEs.

    Open a terminal in this root project directory.

    Create a virtual environment (optional but highly recommended):

    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

    Install dependencies:

    pip install -r requirements.txt

    Install the project in editable mode:

    pip install -e .

You can now open the lattice-isometry-ga folder in your IDE and run experiments/run_ga_search.py directly.
Usage

To run the experiment, simply execute the main script:

python experiments/run_ga_search.py

