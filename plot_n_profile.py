#!/usr/bin/env python3
"""
Compare matter-site <Z_n> for ground state vs 1-meson state |O_{p=0}|Ω⟩.

For each (L, boundary, state) we compute:
  Σ_n  = (-1)^n <Z_n> / 2         (per-site chiral condensate)
  N_n  = (1 - (-1)^n <Z_n>) / 2   (particle excitation density above vacuum)

The 1-meson state has *more* particles than the vacuum — bulk Σ should be
slightly *suppressed*, equivalent to a ~1/L bulk N excess.
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

CSV = Path(__file__).parent / "data" / "n_profile_z2_gauge.csv"
OUT_PNG = Path(__file__).parent / "n_profile_gs_vs_meson.png"
OUT_PDF = Path(__file__).parent / "n_profile_gs_vs_meson.pdf"

df = pd.read_csv(CSV)
df["sign"]  = (-1.0) ** df["matter_site"]
df["sig_n"] = 0.5 * df["sign"] * df["Z_expect"]
df["N_n"]   = 0.5 * (1.0 - df["sign"] * df["Z_expect"])

def bulk_mask(L, n):
    drop = max(1, L // 4)
    return (n >= drop + 1) & (n <= L - drop)
df["in_bulk"] = df.apply(lambda r: bulk_mask(r.L, r.matter_site), axis=1)

agg = (df[df.in_bulk]
       .groupby(["L", "boundary", "state"])
       .agg(sig_bulk=("sig_n", "mean"),
            N_bulk=("N_n", "mean"),
            n_sites=("matter_site", "count"))
       .reset_index())

# meson minus ground excess per state
piv_N = agg.pivot_table(index=["L", "boundary"], columns="state", values="N_bulk").reset_index()
piv_N["delta_N"] = piv_N["meson_p0"] - piv_N["ground"]

# -------- figure --------
fig, axs = plt.subplots(2, 2, figsize=(13, 9.5))

# Panel (a): chiral condensate, both states
ax = axs[0, 0]
for (b, s), g in agg.groupby(["boundary", "state"]):
    label = f"{'OBC' if b=='open_site' else 'PBC'} {'GS' if s=='ground' else 'meson'}"
    ls = "-" if s == "ground" else "--"
    color = "C0" if b == "open_site" else "C1"
    ax.plot(g.L, g.sig_bulk, marker="o", ls=ls, color=color, label=label)
ax.set_xlabel("L")
ax.set_ylabel(r"$\langle \Sigma \rangle_{\rm bulk}$")
ax.set_title("(a) bulk chiral condensate")
ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

# Panel (b): |bulk Σ| of meson state on log scale (exponential decay)
ax = axs[0, 1]
mes = agg[agg.state == "meson_p0"]
for b, g in mes.groupby("boundary"):
    label = "OBC truncate_xz" if b == "open_site" else "PBC drop"
    ax.semilogy(g.L, np.abs(g.sig_bulk), "s-", label=label,
                color=("C0" if b == "open_site" else "C1"))
ax.set_xlabel("L")
ax.set_ylabel(r"$|\langle \Sigma \rangle^{\rm meson}_{\rm bulk}|$")
ax.set_title("(b) meson bulk Σ → 0 exponentially")
ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=9)

# Panel (c): spatial profile, meson state
ax = axs[1, 0]
cmap_o = plt.cm.Blues
cmap_p = plt.cm.Oranges
Ls = sorted(df.L.unique())
for i, L in enumerate(Ls):
    frac = 0.4 + 0.55 * i / max(1, len(Ls) - 1)
    sub = df[(df.L == L) & (df.state == "meson_p0")].sort_values("matter_site")
    for b, g in sub.groupby("boundary"):
        x = g.matter_site / L
        y = (-1) ** g.matter_site * g.Z_expect
        cmap = cmap_o if b == "open_site" else cmap_p
        ls = "-" if b == "open_site" else "--"
        ax.plot(x, y, ls, color=cmap(frac), marker="o", ms=3,
                label=f"L={L} {'OBC' if b=='open_site' else 'PBC'}"
                      if i in (0, len(Ls) - 1) else None)
ax.set_xlabel("matter site n / L")
ax.set_ylabel(r"$(-1)^n \langle Z_n \rangle$  (meson state)")
ax.set_title("(c) meson spatial profile (light → dark = small → large L)")
ax.grid(True, alpha=0.3); ax.legend(fontsize=8, ncol=2, loc="lower center")

# Panel (d): meson − GS profile (excess density at each site)
ax = axs[1, 1]
piv_site = df.pivot_table(index=["L", "boundary", "matter_site"],
                          columns="state", values="N_n").reset_index()
piv_site["delta_N_site"] = piv_site["meson_p0"] - piv_site["ground"]
for i, L in enumerate(Ls):
    frac = 0.4 + 0.55 * i / max(1, len(Ls) - 1)
    sub = piv_site[piv_site.L == L].sort_values("matter_site")
    for b, g in sub.groupby("boundary"):
        x = g.matter_site / L
        y = g.delta_N_site
        cmap = cmap_o if b == "open_site" else cmap_p
        ls = "-" if b == "open_site" else "--"
        ax.plot(x, y, ls, color=cmap(frac), marker="o", ms=3,
                label=f"L={L} {'OBC' if b=='open_site' else 'PBC'}"
                      if i in (0, len(Ls) - 1) else None)
ax.axhline(0, color="k", lw=0.5, ls=":")
ax.set_xlabel("matter site n / L")
ax.set_ylabel(r"$\langle N_n\rangle_{\rm meson} - \langle N_n\rangle_{\rm GS}$")
ax.set_title("(d) per-site particle excess of meson over GS")
ax.grid(True, alpha=0.3); ax.legend(fontsize=8, ncol=2, loc="best")

fig.suptitle(r"Ground state vs 1-meson state $\propto O_{p=0}\,|\Omega\rangle$,"
             r"  $Z_2$ LGT, $m_0{=}0.1$, $\eta{=}0.5$, $\alpha{=}1.0$, bg$=({-}1,{-}1)$",
             fontsize=12)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150)
fig.savefig(OUT_PDF)
print(f"wrote {OUT_PNG}")
print("\nbulk aggregates:")
print(agg.to_string(index=False))
print("\nmeson − GS bulk N excess:")
print(piv_N.to_string(index=False))
