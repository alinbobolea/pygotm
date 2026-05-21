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


def figure_fabm_coupling() -> None:
    """Figure 3: pyGOTM–pyfabm chunked coupling architecture.

    Two-column layout showing the sequential per-chunk execution:
      1. Physics pass (Numba JIT) fills hydro buffers.
      2. pyfabm pass reads buffers step-by-step.
    The biological state cc passes horizontally from chunk k to chunk k+1.
    """
    fig, ax = plt.subplots(figsize=(13.0, 10.5))
    ax.axis("off")
    ax.set_xlim(0, 13.0)
    ax.set_ylim(0, 10.5)

    # ── colour palette ────────────────────────────────────────────────────────
    C_PHYS = "#2c7bb6"
    C_BIO = "#1a9641"
    C_BUF = "#fdae61"
    C_CC = "#7b2f8a"  # biological state continuity
    C_ARROW_PB = "#0571b0"  # physics → buffer
    C_ARROW_BF = "#ca0020"  # buffer → FABM
    GREY = "#555555"

    BOXW = 4.0  # box width for both columns

    # Column centres and left edges
    Lcx, Lx = 2.5, 0.5  # Chunk k   column
    Rcx, Rx = 10.5, 8.5  # Chunk k+1 column

    # Vertical layout (bottom → top)
    # Gap sizes: Buffer→FABM = 2.0 units (holds 4 var lines + header)
    #            Physics→Buffer = 1.2 units (holds 2 label lines)
    fab_y, fab_h = 0.5, 2.5  # pyfabm:  y = 0.5 .. 3.0
    buf_y, buf_h = 5.0, 1.0  # buffers: y = 5.0 .. 6.0   gap = 2.0
    phy_y, phy_h = 7.2, 2.6  # physics: y = 7.2 .. 9.8   gap = 1.2

    cc_y = fab_y + fab_h / 2  # horizontal cc arrow at FABM mid-height = 1.75

    # ── Chunk column headers ──────────────────────────────────────────────────
    for cx, lbl in [(Lcx, "CHUNK  k"), (Rcx, "CHUNK  k+1")]:
        ax.text(
            cx,
            10.0,
            lbl,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#333333",
            bbox=dict(fc="#eeeeee", ec="#aaaaaa", pad=4, boxstyle="round,pad=0.3"),
        )

    # ── Physics boxes ──────────────────────────────────────────────────────────
    for x in (Lx, Rx):
        ax.add_patch(
            plt.Rectangle(
                (x, phy_y), BOXW, phy_h, lw=2, ec=C_PHYS, fc="#d0e4f3", zorder=2
            )
        )
        ax.text(
            x + BOXW / 2,
            phy_y + phy_h - 0.20,
            "pyGOTM Physics Kernel",
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            color=C_PHYS,
        )
        for i, item in enumerate(
            [
                r"Numba JIT  (run_compiled_time_loop)",
                r"• Coriolis, pressure, U/V momentum",
                r"• Temperature, salinity equations",
                r"• Turbulence closure  ($k$–$\varepsilon$, $k$–$\omega$, …)",
                r"• hydro_store=ON  →  fill buffers",
            ]
        ):
            ax.text(
                x + 0.18,
                phy_y + phy_h - 0.52 - i * 0.38,
                item,
                ha="left",
                va="top",
                fontsize=7.5,
                color=GREY,
            )

    # ── Hydro buffer boxes ────────────────────────────────────────────────────
    for x in (Lx, Rx):
        ax.add_patch(
            plt.Rectangle(
                (x, buf_y), BOXW, buf_h, lw=1.5, ec=C_BUF, fc="#fff3cd", zorder=2
            )
        )
        ax.text(
            x + BOXW / 2,
            buf_y + buf_h - 0.16,
            "Hydrodynamic State Buffers",
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            color="#b45309",
        )
        ax.text(
            x + BOXW / 2,
            buf_y + buf_h - 0.44,
            r"T, S, $\rho$, $h$, $\nu_h$, rad, $\tau_b$  ·  shape $(C+1)\times(N_\mathrm{lev}+1)$",
            ha="center",
            va="top",
            fontsize=7.0,
            color=GREY,
        )

    # ── pyfabm boxes ──────────────────────────────────────────────────────────
    for x in (Lx, Rx):
        ax.add_patch(
            plt.Rectangle(
                (x, fab_y), BOXW, fab_h, lw=2, ec=C_BIO, fc="#d4edda", zorder=2
            )
        )
        ax.text(
            x + BOXW / 2,
            fab_y + fab_h - 0.20,
            "pyfabm Engine",
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            color=C_BIO,
        )
        for i, item in enumerate(
            [
                r"Python  (run_fabm_chunk)",
                r"• set_environment(T, S, $\rho$, …)",
                r"• Sinking / rising advection",
                r"• Turbulent diffusion",
                r"• Biogeochemical rates (getRates)",
                r"• cc  +=  dt · rates",
            ]
        ):
            ax.text(
                x + 0.18,
                fab_y + fab_h - 0.50 - i * 0.35,
                item,
                ha="left",
                va="top",
                fontsize=7.5,
                color=GREY,
            )

    # ── Vertical arrows: Physics ↓ Buffers  (same for both columns) ───────────
    # Physics bottom → Buffer top, labelled in the gap
    pb_gap_mid = (phy_y + buf_y + buf_h) / 2  # mid of gap between Physics and Buffer

    for cx in (Lcx, Rcx):
        ax.annotate(
            "",
            xy=(cx, buf_y + buf_h),
            xytext=(cx, phy_y),
            arrowprops=dict(arrowstyle="-|>", color=C_ARROW_PB, lw=2.2),
            zorder=3,
        )
        ax.text(
            cx,
            pb_gap_mid + 0.08,
            "store snapshots  (every step)",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
            color=C_ARROW_PB,
        )
        ax.text(
            cx,
            pb_gap_mid - 0.08,
            r"T, S, $\rho$, $h$, $\nu_h$, rad, $\tau_b$",
            ha="center",
            va="top",
            fontsize=7.0,
            color=C_ARROW_PB,
        )

    # ── Vertical arrows: Buffers ↓ pyfabm  (same for both columns) ────────────
    # Buffer bottom → FABM top, labelled in the gap
    bf_gap_mid = (buf_y + fab_y + fab_h) / 2  # mid of gap between Buffer and FABM

    for cx in (Lcx, Rcx):
        ax.annotate(
            "",
            xy=(cx, fab_y + fab_h),
            xytext=(cx, buf_y),
            arrowprops=dict(arrowstyle="-|>", color=C_ARROW_BF, lw=2.2),
            zorder=3,
        )
        ax.text(
            cx,
            bf_gap_mid + 0.52,
            "per-step inputs to pyfabm",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
            color=C_ARROW_BF,
        )
        for i, line in enumerate(
            [
                r"T, S, $\rho$, $h$  (profiles)",
                r"$\nu_h$  (turbulent diffusivity)",
                r"rad, $\tau_b$  (shortwave, bottom stress)",
                r"u10, v10,  yearday,  precip",
            ]
        ):
            ax.text(
                cx,
                bf_gap_mid + 0.28 - i * 0.26,
                line,
                ha="center",
                va="top",
                fontsize=6.8,
                color=C_ARROW_BF,
            )

    # ── Horizontal cc arrow: chunk k FABM → chunk k+1 FABM ───────────────────
    ax.annotate(
        "",
        xy=(Rx, cc_y),
        xytext=(Lx + BOXW, cc_y),
        arrowprops=dict(arrowstyle="-|>", color=C_CC, lw=2.5),
        zorder=3,
    )
    ax.text(
        (Lx + BOXW + Rx) / 2,
        cc_y + 0.22,
        "cc  (biological state)",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=C_CC,
    )
    ax.text(
        (Lx + BOXW + Rx) / 2,
        cc_y - 0.18,
        r"shape $(n_\mathrm{vars} \times N_\mathrm{lev})$",
        ha="center",
        va="top",
        fontsize=7.5,
        color=C_CC,
    )

    # Dashed continuation arrow beyond chunk k+1
    ax.annotate(
        "",
        xy=(12.85, cc_y),
        xytext=(Rx + BOXW, cc_y),
        arrowprops=dict(arrowstyle="-|>", color=C_CC, lw=1.5, linestyle="dashed"),
        zorder=3,
    )
    ax.text(12.9, cc_y, "…", ha="left", va="center", fontsize=14, color=C_CC)

    # ── Default chunk-size note ───────────────────────────────────────────────
    ax.text(
        6.5,
        0.20,
        r"Default $C = \max\!\left(\lfloor 86400\,\mathrm{s}/\Delta t \rceil,\,1\right)$,"
        r"  rounded up to nearest output interval",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=GREY,
        style="italic",
    )

    ax.set_title(
        "pyGOTM – pyfabm Chunked Interleaved Coupling",
        fontsize=13,
        fontweight="bold",
        pad=4,
    )

    _save(fig, "fabm_coupling.png")


def main() -> None:
    print("Building pyGOTM documentation figures …")
    figure_staggered_grid()
    figure_crank_nicolson()
    figure_fabm_coupling()
    print("Done.")


if __name__ == "__main__":
    main()
