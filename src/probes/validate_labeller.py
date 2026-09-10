"""Validate the family labeller against the HIDDEN TRUTH, and re-score archived runs.

WHY THIS EXISTS (finding F-11). For five sessions this project reported `population.stationary = 0`
and believed it was a statement about the encoder. It was a statement about the probe: the labeller
called any red spectrum "cyclic", so it could not emit "stationary" at all -- not even for the
generator's own stationary amplitude series. The evidence was printed in every `eval.json` as
`labeller_truth_accuracy: 0.7` and neither of us treated it as a blocker.

So: a readout must be validated against known ground truth BEFORE it is allowed to judge a model.
This makes that a one-command habit. Run it after ANY change to `src/probes/family.py`, and before
believing any population number.

A label is a READOUT, not a training target, so re-scoring an archived run needs no retraining --
which is also why a labeller change can silently rewrite the conclusions of every past run.

Run:  conda run -n oceanai python -m src.probes.validate_labeller
      conda run -n oceanai python -m src.probes.validate_labeller --seeds 0 1 2 --runs .tmps/runs/*
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from omegaconf import OmegaConf

from src.data.synthetic import GenConfig, generate_field
from src.probes.family import amp_ratio, label_family, population, series_stats

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _label(x: np.ndarray) -> tuple[str, dict, float]:
    """Label one RAW series. `amp_ratio` must see the raw values -- see family.amp_ratio."""
    ar = amp_ratio(x)
    if x.std() < 1e-12:                      # a perfectly constant series has no spectrum
        st = dict(trend_share=0.0, osc_share=0.0, residual_share=0.0, cyclic_share=0.0,
                  line_frac=0.0, k_pk=0, tau_e=0, gap1=0.0)
    else:
        st = series_stats(x)
    return label_family(st, amp_ratio=ar), st, ar


def check_truth(cfg, seeds) -> bool:
    """Does the labeller reproduce the generator's own family assignment? Must be 10/10."""
    print("=== labeller vs hidden truth ===")
    ok_all = True
    for seed in seeds:
        _, truth = generate_field(GenConfig(**OmegaConf.to_container(cfg.data, resolve=True)),
                                  seed=seed)
        fams = [m["family"] for m in truth["modes"]]
        labs, stats = [], []
        for m in truth["modes"]:
            lab, st, ar = _label(m["amp"])
            labs.append(lab)
            stats.append((st, ar))
        ok = sum(a == b for a, b in zip(fams, labs))
        pop = population(labs)
        ok_all &= (ok == len(fams))
        print(f"  seed {seed}: {ok}/{len(fams)} correct | population "
              f"{pop['stationary']}/{pop['cyclic']}/{pop['chaotic']}"
              + ("" if ok == len(fams) else
                 "  MISMATCH " + str([(i, f, l) for i, (f, l) in enumerate(zip(fams, labs))
                                      if f != l])))
        if seed == seeds[0]:                 # show the descriptors the decision rests on
            print(f"    {'mode':>4} {'family':>10} {'amp_r':>10} {'trend':>6} {'osc':>5} {'resid':>6}")
            for i, (f, (st, ar)) in enumerate(zip(fams, stats)):
                print(f"    {i:>4} {f:>10} {ar:10.3f} {st['trend_share']:6.2f} "
                      f"{st['osc_share']:5.2f} {st['residual_share']:6.2f}")
    print(f"  -> {'PASS' if ok_all else 'FAIL: fix the readout before judging any model'}")
    return ok_all


def rescore(runs) -> None:
    """Re-label the channels of archived runs. Cheap, and the only honest way to compare across
    runs once the labeller has changed."""
    print("\n=== archived runs re-scored under the CURRENT labeller ===")
    for d in runs:
        f = os.path.join(d, "artifacts.npz")
        if not os.path.exists(f):
            continue
        art = np.load(f, allow_pickle=True)
        S = art["S"]
        roles = [str(r) for r in art["roles"]] if "roles" in art.files else []
        if len(roles) != S.shape[1]:          # ladder-off runs store an EMPTY roles array
            roles = ["-"] * S.shape[1]
        labs = [_label(S[:, i])[0] for i in range(S.shape[1])]
        pop = population(labs)
        flat = [i for i in range(S.shape[1]) if amp_ratio(S[:, i]) < 0.05]
        obey = None
        if roles[0] != "-":
            r2f = dict(slow="stationary", cyclic="cyclic", fast="chaotic")
            obey = sum(l == r2f.get(r, "") for l, r in zip(labs, roles)) / len(labs)
        print(f"  {os.path.basename(d)} K={S.shape[1]:>3}: population "
              f"{pop['stationary']}/{pop['cyclic']}/{pop['chaotic']}"
              f"  flat channels {flat if flat else 'NONE'}"
              f"  min amp_ratio {min(amp_ratio(S[:, i]) for i in range(S.shape[1])):.3f}"
              + (f"  role obedience {obey:.2f}" if obey is not None else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--runs", nargs="*", default=None,
                    help="run dirs to re-score (default: all of .tmps/runs/*)")
    ap.add_argument("--config", default=os.path.join(_REPO, "config/config.yaml"))
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    ok = check_truth(cfg, args.seeds)
    runs = args.runs if args.runs is not None else sorted(glob.glob(
        os.path.join(_REPO, ".tmps/runs/*")))
    rescore(runs)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
