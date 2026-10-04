# foil-opt

Airfoil shape optimization with CMA-ES and the [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil) surrogate model.

![Optimized airfoil vs NACA 2412](optimized_airfoil.png)

## How it works

- **Shape:** 8 upper and 8 lower Kulfan (CST) weights, plus a leading-edge weight (17 parameters), starting from a NACA 2412
- **Objective:** maximize mean L/D at α = 2°, 4°, 6° and Re = 5·10⁵
- **Constraints (penalties):** max thickness ≥ 12%, surfaces must not cross, NeuralFoil confidence ≥ 0.9
- **Optimizer:** CMA-ES (`cma`) or Differential Evolution (`scipy`), with each generation scored in one batched NeuralFoil call

## Result

|           | Mean L/D | CL at 4° | CM    |
|-----------|----------|----------|-------|
| NACA 2412 | 79       | 0.70     | −0.05 |
| Optimized | 156      | 1.68     | −0.29 |

> The optimum has very high camber and a large nose-down pitching moment, so it isn't practical yet. Next steps are a CM constraint and checking the result with XFoil.

## Usage

```bash
uv sync
uv run python src/cma_example.py                    # CMA-ES
uv run python src/cma_example.py --optimizer de     # Differential Evolution
```

The run takes a few seconds and writes `optimized_airfoil.png` (CMA-ES) or `optimized_airfoil_de.png` (DE).
