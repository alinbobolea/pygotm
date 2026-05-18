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

    Shows the interleaved physics/biogeochemistry loop: each chunk runs the
    Numba-compiled physics kernel first (storing hydrodynamic snapshots), then
    drives pyfabm through those snapshots.  Variables exchanged at each step
    are shown as labelled arrows between the two engines.
    """
    fig, ax = plt.subplots(figsize=(11.0, 7.5))
    ax.axis("off")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7.5)

    # ── colour palette ────────────────────────────────────────────────────────
    C_PHYS = "#2c7bb6"     # pyGOTM physics box
    C_BIO  = "#1a9641"     # pyfabm box
    C_BUF  = "#fdae61"     # hydro-buffer box
    C_LOOP = "#d7191c"     # chunk-loop annotation
    C_ARROW_PB = "#0571b0" # physics → buffer arrow
    C_ARROW_BF = "#ca0020" # buffer → FABM arrow
    C_ARROW_FB = "#006837" # FABM → physics feedback arrow
    GREY   = "#555555"

    BOXH  = 2.8   # box height
    BOXW  = 3.0   # box width

    # ── Physics box (left) ────────────────────────────────────────────────────
    px, py = 0.6, 2.4
    phys_box = plt.Rectangle(
        (px, py), BOXW, BOXH,
        linewidth=2, edgecolor=C_PHYS, facecolor="#d0e4f3", zorder=2,
    )
    ax.add_patch(phys_box)
    ax.text(
        px + BOXW / 2, py + BOXH - 0.22,
        "pyGOTM Physics Kernel",
        ha="center", va="top", fontsize=11, fontweight="bold", color=C_PHYS,
    )
    phys_items = [
        r"Numba JIT  (run_compiled_time_loop)",
        r"• Coriolis, pressure, U/V momentum",
        r"• Temperature, salinity equations",
        r"• Turbulence closure  ($k$–$\varepsilon$, $k$–$\omega$, …)",
        r"• Air-sea fluxes, ice thermodynamics",
        r"• Stores snapshots → hydro buffers",
    ]
    for i, item in enumerate(phys_items):
        ax.text(
            px + 0.15, py + BOXH - 0.55 - i * 0.37,
            item, ha="left", va="top",
            fontsize=7.8, color=GREY,
            fontfamily="monospace" if "texttt" not in item else "sans-serif",
        )

    # ── Hydro buffer box (centre) ─────────────────────────────────────────────
    bx, by = 3.9, 3.2
    bw, bh = 3.2, 1.8
    buf_box = plt.Rectangle(
        (bx, by), bw, bh,
        linewidth=1.5, edgecolor=C_BUF, facecolor="#fff3cd", zorder=2,
    )
    ax.add_patch(buf_box)
    ax.text(
        bx + bw / 2, by + bh - 0.18,
        "Hydrodynamic State Buffers",
        ha="center", va="top", fontsize=9.5, fontweight="bold", color="#b45309",
    )
    buf_items = [
        r"T, S, $\rho$, $h$, $\nu_h$, rad, $\tau_b$",
        r"shape: $(C+1)\times(N_\mathrm{lev}+1)$",
        r"one row per physics timestep",
    ]
    for i, item in enumerate(buf_items):
        ax.text(
            bx + bw / 2, by + bh - 0.55 - i * 0.37,
            item, ha="center", va="top", fontsize=8, color=GREY,
        )

    # ── FABM box (right) ──────────────────────────────────────────────────────
    fx, fy = 7.4, 2.4
    fabm_box = plt.Rectangle(
        (fx, fy), BOXW, BOXH,
        linewidth=2, edgecolor=C_BIO, facecolor="#d4edda", zorder=2,
    )
    ax.add_patch(fabm_box)
    ax.text(
        fx + BOXW / 2, fy + BOXH - 0.22,
        "pyfabm Engine",
        ha="center", va="top", fontsize=11, fontweight="bold", color=C_BIO,
    )
    fabm_items = [
        r"Python  (run_fabm_chunk)",
        r"• Set environment (T, S, ρ, PAR, …)",
        r"• Sinking/rising advection (P2-PDM)",
        r"• Turbulent diffusion (diff_center)",
        r"• Biogeochemical rates (getRates)",
        r"• Apply source/sink: cc += dt·rates",
    ]
    for i, item in enumerate(fabm_items):
        ax.text(
            fx + 0.15, fy + BOXH - 0.55 - i * 0.37,
            item, ha="left", va="top",
            fontsize=7.8, color=GREY,
        )

    # ── Arrow: Physics → Buffer ───────────────────────────────────────────────
    ax.annotate(
        "",
        xy=(bx, by + bh * 0.65),
        xytext=(px + BOXW, py + BOXH * 0.65),
        arrowprops=dict(arrowstyle="-|>", color=C_ARROW_PB, lw=2.0),
        zorder=3,
    )
    ax.text(
        (px + BOXW + bx) / 2, py + BOXH * 0.65 + 0.18,
        "store snapshots\nevery step",
        ha="center", va="bottom", fontsize=7.5, color=C_ARROW_PB,
    )

    # ── Arrow: Buffer → FABM ──────────────────────────────────────────────────
    ax.annotate(
        "",
        xy=(fx, fy + BOXH * 0.5),
        xytext=(bx + bw, by + bh * 0.35),
        arrowprops=dict(arrowstyle="-|>", color=C_ARROW_BF, lw=2.0),
        zorder=3,
    )
    buf_labels = [
        "T, S, ρ, h  (profiles)",
        "nuh  (turbulent diffusivity)",
        "rad  (shortwave)",
        "τ_b  (bottom stress)",
        "u10, v10, yearday, precip",
    ]
    bfx = (bx + bw + fx) / 2
    bfy = (by + bh * 0.35 + fy + BOXH * 0.5) / 2
    for i, lbl in enumerate(buf_labels):
        ax.text(
            bfx, bfy + 0.25 - i * 0.28,
            lbl, ha="center", va="center",
            fontsize=6.8, color=C_ARROW_BF,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5),
        )

    # ── Arrow: FABM feedback → Physics (light attenuation) ───────────────────
    ax.annotate(
        "",
        xy=(px + BOXW * 0.7, py),
        xytext=(fx + BOXW * 0.3, fy),
        arrowprops=dict(
            arrowstyle="-|>", color=C_ARROW_FB, lw=1.6, linestyle="dashed",
        ),
        zorder=3,
    )
    ax.text(
        (px + BOXW * 0.7 + fx + BOXW * 0.3) / 2,
        py - 0.22,
        "kc (bio-shading feedback, optional)",
        ha="center", va="top", fontsize=7.5, color=C_ARROW_FB, style="italic",
    )

    # ── Chunk loop annotation ─────────────────────────────────────────────────
    loop_y = 7.1
    ax.annotate(
        "",
        xy=(10.6, loop_y),
        xytext=(0.4, loop_y),
        arrowprops=dict(arrowstyle="-|>", color=C_LOOP, lw=1.8),
    )
    ax.text(
        5.5, loop_y + 0.18,
        r"Chunk loop  ($k = 0, 1, 2, \dots$,  each chunk $= C$ physics timesteps)",
        ha="center", va="bottom", fontsize=9, color=C_LOOP, fontweight="bold",
    )
    # Chunk-boundary ticks
    for frac in (0.0, 0.33, 0.66, 1.0):
        xp = 0.4 + frac * (10.6 - 0.4)
        ax.plot([xp, xp], [loop_y - 0.12, loop_y + 0.12], color=C_LOOP, lw=1.5)
        if frac < 1.0:
            ax.text(
                xp + (10.6 - 0.4) * 0.165, loop_y - 0.32,
                f"Chunk {int(frac / 0.33)}",
                ha="center", va="top", fontsize=7.5, color=C_LOOP,
            )

    # ── Chunk-size annotation ─────────────────────────────────────────────────
    ax.text(
        5.5, 0.35,
        r"Default $C = \max\!\left(\lfloor 86400\,\mathrm{s}/\Delta t \rceil,\,1\right)$,"
        r"  rounded up to nearest output interval",
        ha="center", va="bottom", fontsize=8.5, color=GREY, style="italic",
    )

    # ── State continuity arrow ────────────────────────────────────────────────
    ax.annotate(
        "",
        xy=(fx + BOXW / 2, fy + BOXH + 0.55),
        xytext=(fx + BOXW / 2, fy + BOXH),
        arrowprops=dict(arrowstyle="-|>", color=C_BIO, lw=1.4),
    )
    ax.text(
        fx + BOXW / 2, fy + BOXH + 0.6,
        "cc passed to next chunk",
        ha="center", va="bottom", fontsize=7.5, color=C_BIO,
    )

    ax.set_title(
        "pyGOTM – pyfabm Chunked Interleaved Coupling",
        fontsize=13, fontweight="bold", pad=4,
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
