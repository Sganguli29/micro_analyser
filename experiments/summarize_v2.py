"""experiments/summarize_v2.py - Aggregates results/v2* into summary tables, significance tests and figures."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"


def ms(x, d=4):
    return pd.Series({"mean": round(x.mean(), d), "sd": round(x.std(ddof=1), d) if len(x) > 1 else 0.0})


def agg_configs(d):
    m = ["accuracy", "macro_f1", "mean_cost_ms", "mean_blocks", "cost_saving_pct", "rejected_pct"]
    g = d.groupby("configuration")[m]
    out = g.mean().round(4).add_suffix("_mean").join(g.std(ddof=1).round(4).add_suffix("_sd"))
    if "param" in d:
        out = out.join(d.groupby("configuration")["param"].agg(lambda s: ",".join(str(v) for v in s if v == v)))
    if "w_context" in d:
        out = out.join(d.groupby("configuration")["w_context"].mean().round(4))
    if "context_only_acc" in d:
        out = out.join(d.groupby("configuration")["context_only_acc"].mean().round(4))
    return out


def mcnemar(y, a, b):
    ca, cb = a == y, b == y
    n01 = int(np.sum(ca & ~cb)); n10 = int(np.sum(~ca & cb))
    stat = (abs(n01 - n10) - 1) ** 2 / max(n01 + n10, 1)
    return stat, float(chi2.sf(stat, 1)), n01, n10


def boot(y, a, b, ca, cb, n=10000, seed=0):
    rs = np.random.RandomState(seed)
    N = len(y); da = (a == y).astype(float) - (b == y).astype(float); dc = ca - cb
    idx = rs.randint(0, N, size=(n, N))
    acc = da[idx].mean(1); cost = dc[idx].mean(1)
    return (da.mean(), np.percentile(acc, 2.5), np.percentile(acc, 97.5),
            dc.mean(), np.percentile(cost, 2.5), np.percentile(cost, 97.5))


def significance(out, pairs):
    files = sorted(out.glob("preds_seed*.npz"))
    rows = []
    for A, B in pairs:
        ys, pa, pb, ca, cb, ps = [], [], [], [], [], []
        for f in files:
            z = np.load(f)
            y = z["y"]; a = z[f"{A}__pred"]; b = z[f"{B}__pred"]
            ys.append(y); pa.append(a); pb.append(b); ca.append(z[f"{A}__cost"]); cb.append(z[f"{B}__cost"])
            ps.append(mcnemar(y, a, b)[1])
        y, a, b = np.concatenate(ys), np.concatenate(pa), np.concatenate(pb)
        st, p, n01, n10 = mcnemar(y, a, b)
        bt = boot(y, a, b, np.concatenate(ca), np.concatenate(cb))
        rows.append(dict(a=A, b=B, n=len(y), d_acc_pts=100 * bt[0], d_acc_lo=100 * bt[1], d_acc_hi=100 * bt[2],
                         mcnemar_p_pooled=p, mcnemar_p_seed_min=min(ps), mcnemar_p_seed_max=max(ps),
                         a_only_correct=n01, b_only_correct=n10,
                         d_cost_ms=bt[3], d_cost_lo=bt[4], d_cost_hi=bt[5]))
    return pd.DataFrame(rows)


def tradeoff_fig(out, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = pd.read_csv(out / "tradeoff.csv")
    g = c.groupby(["policy", "param"])[["accuracy", "mean_cost_ms"]].mean().reset_index()
    full = pd.read_csv(out / "configs.csv")
    fc = full[full.configuration == "ref_all_blocks_glob"]
    plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 8})
    fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=300)
    series = [("fixed_tau_glob", "Fixed τ, class prior", "#2a78d6", "o"),
              ("fixed_tau_ctx", "Fixed τ, context prior", "#eb6834", "s"),
              ("learned_random_forest", "Learned stopping (RF)", "#1baf7a", "^")]
    for key, lab, col, mk in series:
        s = g[g.policy == key].sort_values("mean_cost_ms")
        ax.plot(s.mean_cost_ms, 100 * s.accuracy, color=col, lw=1.6, marker=mk, ms=4.5,
                markeredgecolor="white", markeredgewidth=0.6, label=lab, zorder=3)
    ax.axhline(100 * fc.accuracy.mean(), color="#8a8984", lw=0.9, ls="--", zorder=1)
    ax.axvline(fc.mean_cost_ms.mean(), color="#8a8984", lw=0.9, ls=":", zorder=1)
    ax.text(fc.mean_cost_ms.mean() - 0.1, 100 * fc.accuracy.mean() - 0.6, "all five blocks",
            ha="right", va="top", fontsize=7, color="#52514e")
    ax.set_xlabel("Mean extraction cost per particle (ms)")
    ax.set_ylabel("Test accuracy (%)")
    ax.grid(True, color="#e6e5e0", lw=0.6, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#8a8984")
    ax.tick_params(colors="#52514e", length=2)
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    lo = 100 * g[g.policy.isin([s[0] for s in series])].accuracy.min()
    ax.set_ylim(max(lo - 1, 70), 100 * fc.accuracy.mean() + 1.2)
    fig.tight_layout()
    fig.savefig(fig_dir / "tradeoff.png", dpi=300)
    fig.savefig(fig_dir / "tradeoff.pdf")
    return g


def main():
    out = R / "v2"
    S = {}
    cfg = pd.read_csv(out / "configs.csv")
    S["configs"] = agg_configs(cfg)
    S["configs"].to_csv(out / "summary_configs.csv")
    for v in ("v2_allmask", "v2_bboxshape"):
        if (R / v / "configs.csv").exists():
            agg_configs(pd.read_csv(R / v / "configs.csv")).to_csv(R / v / "summary_configs.csv")
    st = pd.read_csv(out / "stopping.csv")
    st.groupby("model")[["threshold", "accuracy", "f1", "auc", "recall", "specificity", "stop_label_rate"]] \
        .agg(["mean", "std"]).round(4).to_csv(out / "summary_stopping.csv")
    bl = pd.read_csv(out / "blocks.csv")
    bl.groupby("block")[["test_acc", "w_glob", "w_ctx"]].agg(["mean", "std"]).round(4).to_csv(out / "summary_blocks.csv")
    pairs = [("2_context_fixed085", "1_global_fixed085"),
             ("3_context_learned_rf", "1T_global_tuned_tau"),
             ("3_context_learned_rf", "2T_context_tuned_tau"),
             ("3_context_learned_rf", "1_global_fixed085"),
             ("1T_global_tuned_tau", "1_global_fixed085")]
    significance(out, pairs).round(5).to_csv(out / "significance_v2.csv", index=False)
    if (out / "lopo_detectors.csv").exists():
        lo = pd.read_csv(out / "lopo_detectors.csv")
        m = ["auroc", "aupr", "fpr95", "unknown_detect", "false_unknown"]
        lo.groupby("method")[m].agg(["mean", "std"]).round(4).to_csv(out / "summary_lopo_methods.csv")
        lo[lo.method == "dual_rule"].groupby("held_out")[["closed_acc", "auroc", "unknown_detect", "false_unknown"]] \
            .agg(["mean", "std"]).round(4).to_csv(out / "summary_lopo_dual_by_polymer.csv")
        lo[lo.method == "energy"].groupby("held_out")[["auroc", "unknown_detect", "false_unknown"]] \
            .agg(["mean", "std"]).round(4).to_csv(out / "summary_lopo_energy_by_polymer.csv")
        best = lo.groupby(["held_out", "method"]).auroc.mean().reset_index()
        best.loc[best.groupby("held_out").auroc.idxmax()].to_csv(out / "summary_lopo_best_by_polymer.csv", index=False)
    if (out / "lopo_pipeline.csv").exists():
        lp = pd.read_csv(out / "lopo_pipeline.csv")
        m = ["unknown_detect", "false_unknown", "accepted_acc", "mean_cost_ms", "mean_blocks"]
        lp.groupby("method")[m].agg(["mean", "std"]).round(4).to_csv(out / "summary_lopo_pipeline.csv")
        lp.groupby(["method", "held_out"])[m].mean().round(4).to_csv(out / "summary_lopo_pipeline_by_polymer.csv")
    fig_dir = R / "v2" / "figures"; fig_dir.mkdir(exist_ok=True)
    tradeoff_fig(out, fig_dir).round(4).to_csv(out / "summary_tradeoff.csv", index=False)
    print(S["configs"].to_string())


if __name__ == "__main__":
    main()
