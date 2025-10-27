
---

# AMFD for QUBO

## Overview

The **QUBO (Quadratic Unconstrained Binary Optimization)** machine is a general-purpose solver capable of efficiently finding approximate solutions to a wide range of combinatorial optimization problems.
This repository provides a **PyTorch implementation** of the **Annealed Mean-Field Descent (AMFD)** algorithm, along with demonstration code and sample problem formulations.

```bash
.
├── generator.py       # Functions for QUBO formulation of representative problems
├── mediator.py        # Conversion between formulated functions and QUBO matrices / solution reconstruction
├── read_file.py       # Load benchmark dataset file
├── datasets
│       ├── gcp
│       ├── mcp
│       ├── misp
│       ├── qap
│       └── tsp
├── amfd 
│       ├── main.py
│       └── amfd_optimizer.py # AMFD optimization algorithm
├── gurobi
│       ├── main.py
│       └── gurobi_optimizer.py
└── README.md          # This file
```

### File Descriptions

* **`generator.py`**
  Contains functions to define typical combinatorial optimization problems in QUBO form.

* **`mediator.py`**
  Provides utilities to convert the formulated problems into QUBO matrices and to reconstruct the original solution shape from QUBO outputs.

---

## Clone the Repository

```bash
git clone https://github.com/kyo-kuroki/AMFD.git
cd AMFD/amfd
```

---

## Evaluation

To run a basic evaluation using the AMFD optimizer:

```bash
python main.py
```

---
