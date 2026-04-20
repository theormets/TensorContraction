# Accelerating Tensor Operations in Computational Solid Mechanics

**Optimised contraction strategies and vectorised execution from deep learning, applied to mechanics**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/tensor-contraction-mechanics/blob/main/tensor_mechanics_benchmark.ipynb)

---

## Overview

This repository contains the complete benchmark code and publication figures for the paper:

> **Accelerating tensor operations in computational solid mechanics through optimised contraction strategies and vectorised execution: a preliminary benchmark study**  
> P G Kubendran Amos  
> *Department of Metallurgical and Materials Engineering, National Institute of Technology Tiruchirappalli*

The study demonstrates that three techniques from the deep learning software ecosystem — **contraction path optimisation**, **vectorised batch execution (`vmap`)**, and **just-in-time compilation (`jit`)** — can reduce the computational cost of tensor operations in solid mechanics by **one to two orders of magnitude on a single CPU core**, without GPU hardware.

## Key Results

| Benchmark | Operation | Speedup | Mechanism |
|-----------|-----------|:-------:|-----------|
| Anisotropic rotation | $C'_{pqrs} = R_{pi} R_{qj} C_{ijkl} R_{rk} R_{sl}$ | **377×** | Path opt + `vmap`/`jit` |
| J2 plasticity tangent | Algorithmic tangent $\mathbb{C}_{ep}$ | **181×** | Branchless `vmap`/`jit` |
| High-order fused einsum | $\mathbf{f}_e = \sum_g \mathbf{B}_g^T \mathbf{D} \mathbf{B}_g \mathbf{u}_e$ | **29×** | Fused contraction path |
| High-order sum-factorisation | Tensor-product 1D ops | **8×** | $O(p^6) \to O(p^4)$ |

<p align="center">
  <img src="figures/fig_summary_speedups.pdf" width="600">
</p>

## Repository Structure

```
.
├── tensor_mechanics_benchmark.ipynb   # Main notebook (documented, Colab-ready)
├── figures/
│   ├── fig_case2_rotation.pdf         # Fig 1: Anisotropic rotation results
│   ├── fig_case3_j2plasticity.pdf     # Fig 2: J2 plasticity tangent results
│   ├── fig_case4_highorder.pdf        # Fig 3: High-order matrix-free results
│   └── fig_summary_speedups.pdf       # Fig 4: Summary speedup comparison
├── README.md
├── LICENSE
└── .gitignore
```

## Quick Start

### Option 1: Google Colab (recommended)
Click the **Open in Colab** badge above, then **Runtime → Run all**.  
No installation required. Runs in ~1–2 minutes.

### Option 2: Local execution
```bash
git clone https://github.com/YOUR_USERNAME/tensor-contraction-mechanics.git
cd tensor-contraction-mechanics
pip install numpy jax jaxlib opt-einsum matplotlib
jupyter notebook tensor_mechanics_benchmark.ipynb
```

## Dependencies

| Package | Purpose | Version tested |
|---------|---------|---------------|
| NumPy | Baseline array operations | ≥1.24 |
| JAX | `vmap`, `jit`, XLA compilation | ≥0.4 |
| opt_einsum | Optimal contraction path finding | ≥3.3 |
| matplotlib | Publication figures | ≥3.7 |

All packages are auto-installed by the notebook if not present.

## Benchmarks Explained

### 1. Anisotropic Material Rotation
Rotates a fourth-order triclinic elasticity tensor through 10,000 unique crystal orientations. Demonstrates how contraction path reordering reduces the operation from O(N⁸) to O(N⁵), and how `vmap` eliminates the Python loop to compound the gain.

### 2. J2 Plasticity Tangent Modulus
Computes the algorithmic consistent tangent at 10,000 integration points with mixed elastic/plastic states. Demonstrates **branchless vectorisation**: replacing `if/else` yield checks with `jnp.where` enables SIMD execution and XLA fusion, giving the largest speedup (~181×).

### 3. High-Order Matrix-Free Operator (p=4 Hex)
Applies the element stiffness operator without assembling the global matrix on a p=4 hexahedral mesh (125 nodes, 375 DOF per element, 3,000 elements). Compares dense fused einsum against sum-factorisation, which exploits the tensor-product structure of hexahedral shape functions.

## Citation

If this work is useful for your research, please cite:

```bibtex
@article{amos2025tensor,
  title={Accelerating tensor operations in computational solid mechanics through 
         optimised contraction strategies and vectorised execution: 
         a preliminary benchmark study},
  author={Amos, P G Kubendran},
  journal={Computer Physics Communications},
  year={2025},
  note={Submitted}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Contact

**P G Kubendran Amos**  
Theoretical Metallurgical Group  
Department of Metallurgical and Materials Engineering  
National Institute of Technology Tiruchirappalli  
Tamil Nadu, India 620015  
Email: prince@nitt.edu
