"""Command line: foil-opt {run,compare,pareto,validate}."""

import argparse
import warnings

import numpy as np

from . import config, experiments, plotting, report
from .geometry import baseline, baseline_genome, to_airfoil
from .optimizers import OPTIMIZERS
from .problem import evaluate

NAMES = {key: name for key, (name, _) in OPTIMIZERS.items()}
EVOLUTIONARY = [key for key in OPTIMIZERS if key != "grad"]


def cmd_run(args) -> None:
    report.print_design("NACA 2412", baseline_genome())
    print(f"\nOptimizer: {NAMES[args.optimizer]}, seed {args.seed}, budget {args.budget}")
    result = experiments.run(args.optimizer, args.seed, args.budget, verbose=True)
    experiments.save(result)
    print(f"\n{result.evals} evaluations in {result.seconds:.1f} s")
    report.print_design(NAMES[args.optimizer], result.x)
    plotting.plot_shapes({NAMES[args.optimizer]: to_airfoil(result.x)}, baseline,
                         config.ASSETS_DIR / f"shape_{args.optimizer}.png")


def cmd_compare(args) -> None:
    results = {}
    for key in args.optimizers:
        seeds = [1] if key == "grad" else range(1, args.seeds + 1)
        results[key] = []
        for seed in seeds:
            result = experiments.run(key, seed, args.budget)
            experiments.save(result)
            results[key].append(result)
            print(f"{NAMES[key]:24s} seed {seed}  {result.evals:6d} evals  {result.seconds:5.1f} s  "
                  f"mean CD {result.cost:.5f}")

    print("\n" + report.comparison_table(results, NAMES))

    histories = {NAMES[k]: [np.array(r.history) for r in runs] for k, runs in results.items() if k != "grad"}
    gradient_cd = results["grad"][0].cost if "grad" in results else None
    plotting.plot_convergence(histories, gradient_cd, config.ASSETS_DIR / "convergence.png")
    designs = {NAMES[k]: to_airfoil(experiments.best(runs).x) for k, runs in results.items()}
    plotting.plot_shapes(designs, baseline, config.ASSETS_DIR / "shapes.png")


def cmd_pareto(args) -> None:
    from .pareto import run_nsga2

    front = run_nsga2(baseline_genome(), args.seed)
    print(f"{len(front.mean_cd)} designs on the front")
    print("  mean CD   max |CM|")
    for cd, cm in zip(front.mean_cd, front.max_abs_cm):
        print(f"  {cd:.5f}  {cm:.3f}")

    picks = {"lowest drag": 0, "middle": len(front.mean_cd) // 2, "lowest |CM|": -1}
    designs = {f"{label} (|CM| {front.max_abs_cm[i]:.3f})": to_airfoil(front.genomes[i]) for label, i in picks.items()}
    base = evaluate([baseline_genome()])
    marked = {"NACA 2412": (abs(base.aero["CM"][0]).max(), base.objective[0])}
    cma_runs = experiments.load("cma")
    if cma_runs:
        cma = evaluate([experiments.best(cma_runs).x])
        marked["CMA-ES (CM ≥ −0.1)"] = (abs(cma.aero["CM"][0]).max(), cma.objective[0])
    plotting.plot_pareto(front.mean_cd, front.max_abs_cm, designs, marked, baseline, config.ASSETS_DIR / "pareto.png")


def cmd_validate(args) -> None:
    from .validation import find_xfoil, validate

    xfoil = find_xfoil()
    if xfoil is None:
        print("XFoil not found: install it or set XFOIL to the binary's path.")
        return
    print("Optimizer               CL      Re   CD NeuralFoil  CD XFoil   diff    CM NF  CM XFoil")
    for key in args.optimizers:
        runs = experiments.load(key)
        if not runs:
            print(f"{NAMES[key]}: no results, run `foil-opt run {key}` first")
            continue
        x = experiments.best(runs).x
        c = validate(x, to_airfoil(x), xfoil)
        for i in range(len(c.cl)):
            diff = (c.cd_neuralfoil[i] - c.cd_xfoil[i]) / c.cd_xfoil[i]
            print(f"{NAMES[key]:22s} {c.cl[i]:4.2f}  {c.re[i]:6.0e}   {c.cd_neuralfoil[i]:.5f}     "
                  f"{c.cd_xfoil[i]:.5f}  {diff:+6.1%}  {c.cm_neuralfoil[i]:+.3f}  {c.cm_xfoil[i]:+.3f}")


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    parser = argparse.ArgumentParser(prog="foil-opt", description=__doc__)
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("run", help="run one optimizer")
    p.add_argument("optimizer", choices=OPTIMIZERS)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--budget", type=int, default=config.DEFAULT_BUDGET)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("compare", help="run optimizers over several seeds and plot convergence")
    p.add_argument("--optimizers", nargs="+", choices=OPTIMIZERS, default=list(OPTIMIZERS))
    p.add_argument("--seeds", type=int, default=config.DEFAULT_SEEDS)
    p.add_argument("--budget", type=int, default=config.DEFAULT_BUDGET)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("pareto", help="NSGA-II front of mean CD vs |CM|")
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(func=cmd_pareto)

    p = sub.add_parser("validate", help="compare saved best designs against XFoil")
    p.add_argument("--optimizers", nargs="+", choices=OPTIMIZERS, default=list(OPTIMIZERS))
    p.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    args.func(args)
