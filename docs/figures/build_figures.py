#!/usr/bin/env python3
"""Generate GOTM manual figures for the pyGOTM documentation.

Figures reproduced from: Umlauf, Burchard & Bolding, GOTM User Manual v4.0,
Section 3.1.2, page 25.

Usage
-----
    conda activate pygotm
    python docs/figures/build_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).parent.resolve()


def _save(fig: plt.Figure, name: str) -> Path:
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def figure_staggered_grid(nlev: int = 6) -> None:
    """Figure 1: Staggered vertical grid (GOTM manual Fig. 1, p.25).

    Turbulent quantities (k, ε, ν_t) live at cell interfaces i=0..N.
    Mean-flow quantities (U, V, θ, S) live at cell centres i=1..N.
    """
    fig, ax = plt.subplots(figsize=(6.0, 7.5))
    ax.axis("off")

    x_axis = 0.30

    def y(i: int) -> float:
        return i / nlev

    # Vertical axis line
    ax.annotate(
        "",
        xy=(x_axis, 1.09),
        xytext=(x_axis, -0.03),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2),
        annotation_clip=False,
    )
    ax.text(x_axis, 1.15, r"$z$", ha="center", va="bottom", fontsize=12)

    # Interfaces: filled circles (turbulent quantities)
    # White bbox prevents dashed boundary lines from bleeding through labels
    _lbbox = dict(facecolor="white", edgecolor="none", pad=1.5)
    for i in range(nlev + 1):
        yi = y(i)
        ax.plot(x_axis, yi, "ko", ms=8, zorder=4)
        if i == 0:
            label = r"$i=0$"
        elif i == nlev:
            label = rf"$i=N={nlev}$"
        else:
            label = rf"$i={i}$"
        ax.text(
            x_axis - 0.09, yi, label, ha="right", va="center", fontsize=9, bbox=_lbbox
        )

    # Layer centres: open squares (mean-flow quantities)
    for i in range(1, nlev + 1):
        yc = (y(i - 1) + y(i)) / 2.0
        ax.plot(
            x_axis, yc, "ws", ms=10, markeredgecolor="k", markeredgewidth=1.3, zorder=3
        )

    # Layer thickness arrow for one representative layer
    mid = nlev // 2
    x_arrow = x_axis + 0.22
    ax.annotate(
        "",
        xy=(x_arrow, y(mid)),
        xytext=(x_arrow, y(mid - 1)),
        arrowprops=dict(arrowstyle="<->", color="dimgray", lw=1.0),
    )
    ax.text(
        x_arrow + 0.04,
        (y(mid) + y(mid - 1)) / 2,
        r"$h_i$",
        va="center",
        fontsize=10,
        color="dimgray",
    )

    # Surface label — above the dashed line, left-aligned to avoid right clipping
    ax.text(
        x_axis,
        y(nlev) + 0.07,
        r"$z = \zeta(t)$  (surface)",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="steelblue",
    )
    # Bottom label — well below dashed line so legend doesn't overlap it
    ax.text(
        x_axis,
        y(0) - 0.10,
        r"$z = -H$  (bottom)",
        ha="center",
        va="top",
        fontsize=9.5,
        color="saddlebrown",
    )

    # Horizontal dashed lines — start right of labels, end before right edge
    # xlim span = 1.10; x=0.22 (right edge of labels) → fraction ≈ 0.38;
    # xlim=(-0.20, 1.05) span=1.25; x_axis=0.30 → 0.40; x=0.70 → 0.72
    ax.axhline(
        y(nlev), xmin=0.40, xmax=0.72, color="steelblue", lw=0.8, ls="--", zorder=1
    )
    ax.axhline(
        y(0), xmin=0.40, xmax=0.72, color="saddlebrown", lw=0.8, ls="--", zorder=1
    )

    # Legend — upper-right quadrant, clear of all data and labels
    iface_handle = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="black",
        markersize=8,
        label=r"interfaces $i=0,\dots,N$: $k,\,\varepsilon,\,\nu_t,\,\kappa_t$",
    )
    centre_handle = plt.Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1.3,
        markersize=9,
        label=r"centres $i=1,\dots,N$: $U,\,V,\,\theta,\,S$",
    )
    # grid spans y=[0,1]; center in axes fraction = (0.5+0.32)/1.77 ≈ 0.463
    ax.legend(
        handles=[iface_handle, centre_handle],
        loc="center right",
        bbox_to_anchor=(1.0, 0.62),
        fontsize=8.5,
        framealpha=0.95,
    )

    ax.set_xlim(-0.20, 1.05)
    ax.set_ylim(-0.32, 1.45)
    ax.set_title("Staggered Vertical Grid", fontsize=12, fontweight="bold", pad=8)

    _save(fig, "staggered_grid.png")


def figure_crank_nicolson(sigma: float = 0.6) -> None:
    """Figure 2: Crank–Nicolson time stepping (GOTM manual Fig. 2, p.25).

    Shows the four time levels used in the scheme and their roles.
    The manual illustrates sigma=0.6; pyGOTM uses sigma=1 (fully implicit).
    """
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.axis("off")

    T = 0.08
    DT = 0.76
    ts = [T, T + (1 - sigma) * DT, T + sigma * DT, T + DT]
    labels_top = [
        r"$T$",
        r"$T + (1-\sigma)\Delta t$",
        r"$T + \sigma\Delta t$",
        r"$T + \Delta t$",
    ]
    labels_bot = [
        r"$u$" "\n(known)",
        r"$\tilde{u}$" "\n(intermediate)",
        r"$\bar{u}$" "\n(CN average)",
        r"$\hat{u}$" "\n(solution)",
    ]
    colors = ["steelblue", "gray", "darkorange", "seagreen"]

    y_axis = 0.50

    # Horizontal time axis arrow
    ax.annotate(
        "",
        xy=(1.0, y_axis),
        xytext=(0.0, y_axis),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.3),
        annotation_clip=False,
    )
    ax.text(1.02, y_axis, r"$t$", va="center", fontsize=12)

    for x, lt, lb, col in zip(ts, labels_top, labels_bot, colors, strict=True):
        # Tick
        ax.plot([x, x], [y_axis - 0.05, y_axis + 0.05], color="black", lw=1.0, zorder=2)
        # Dot on axis
        ax.plot(x, y_axis, "o", color=col, ms=11, zorder=3)
        # Label above
        ax.text(
            x, y_axis + 0.15, lt, ha="center", va="bottom", fontsize=8.5, color="black"
        )
        # Label below
        ax.text(x, y_axis - 0.15, lb, ha="center", va="top", fontsize=8, color=col)

    # Δt brace
    ax.annotate(
        "",
        xy=(ts[3], y_axis + 0.32),
        xytext=(ts[0], y_axis + 0.32),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.0),
    )
    ax.text(
        (ts[0] + ts[3]) / 2,
        y_axis + 0.36,
        r"$\Delta t$",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    # sigma note
    ax.text(
        0.5,
        0.04,
        (
            rf"Crank–Nicolson parameter: $\sigma = {sigma}$ "
            r"(pyGOTM uses $\sigma = 1$, fully implicit)"
        ),
        ha="center",
        va="bottom",
        fontsize=8,
        style="italic",
        transform=ax.transAxes,
        color="dimgray",
    )

    ax.set_xlim(-0.03, 1.05)
    ax.set_ylim(0.0, 1.08)
    ax.set_title("Crank–Nicolson Time Stepping", fontsize=11, fontweight="bold", pad=6)

    _save(fig, "crank_nicolson.png")


def main() -> None:
    print("Building pyGOTM documentation figures …")
    figure_staggered_grid()
    figure_crank_nicolson()
    print("Done.")


if __name__ == "__main__":
    main()
