import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Diagnostic plot: tau(M) (the running integrated-autocorrelation-time
# estimate) as a function of the lag/window M, for every L in the
# simulation -- so you can *look* at where the curve actually flattens out
# instead of just trusting tau_int_fft's automatic windowing blindly.
#
# Sokal's criterion (M >= c*tau(M), c=5 by default) usually picks a
# sensible M on its own, but "usually" isn't "always": for a noisy or
# marginally-converged run the automatic M can land a bit early (still
# trending) or on a jittery point rather than a clean plateau.
#
# Deliberately standalone: unlike the rest of analysis/, this script does
# NOT import from functions.py -- tau_running()/sokal_window() below are
# self-contained copies of the same computation tau_int_fft() uses there,
# so this file has no dependency on the rest of the project and can be
# copied/run on its own. The tradeoff is the one this avoids elsewhere: if
# functions.py's autocorrelation math ever changes, these copies won't
# pick it up automatically and need to be updated here too.
#
# Input convention matches the rest of analysis/ regardless: sim_name +
# alg, and metadata.csv (written by the C simulation) as the reference for
# which L and beta values exist -- nothing here is guessed or re-derived.


def tau_running(x):
    """
    Running integrated-autocorrelation-time estimate tau(M) for every
    window M = 0, 1, ..., N-1 (plus the underlying normalized
    autocorrelation function rho), computed via FFT.
    """
    x = np.asarray(x)
    N = len(x)

    x = x - np.mean(x)

    nfft = 2**int(np.ceil(np.log2(2*N)))

    f = np.fft.fft(x, n=nfft)
    power = f * np.conjugate(f)

    acov = np.fft.ifft(power).real[:N]
    acov /= N # keeps only the first 0,... N-1 elements (on nfft)
    # biased convention used; acov/=N works better than un-biased acov/=N-lag... (biased+more stable for large k)

    rho = acov / acov[0]

    taus = np.concatenate(([0.5], 0.5 + np.cumsum(rho[1:])))
    return taus, rho


def sokal_window(taus, N, c=5, min_window=4):
    """
    Sokal's automatic windowing: the smallest window M for which
    M >= c*tau(M) (self-consistent stopping criterion), subject to a
    minimum window and to M staying below N/2. Returns the chosen M (an
    index into taus), or None if no M satisfies the criterion within N/2.
    """
    idx = np.arange(0, len(taus)) # lag k
    valid = (idx >= min_window) & (idx < N // 2) & (idx >= c * taus)
    if not np.any(valid):
        return None
    return int(idx[np.argmax(valid)])


os.makedirs("results/figure", exist_ok=True)

beta_c = 0.5 * np.log(1 + np.sqrt(2))

sim_name = input("Simulation name: ").strip()
alg = input("Algorithm (metropolis/wolff): ").strip()

filename = f"data/{sim_name}/metadata.csv"
metadata = pd.read_csv(filename)

# Which beta to inspect for each L. Defaults to the beta closest to
# beta_c -- same convention dynamic_z.py uses -- since that's where tau is
# largest and the automatic windowing is most likely to be marginal. If
# you want a *different* beta for a given L, just edit BETA_OVERRIDE
# below, e.g. BETA_OVERRIDE = {64: 0.44} to inspect L=64 at beta=0.44
# instead of whichever simulated beta happens to be closest to beta_c.
BETA_OVERRIDE = {}

MAX_LOCAL_MULT = 6 if alg == "wolff" else 1.2

# style -- identical rcParams block to dynamic_z.py/wham.py/binder.py/
# plot.py/hist.py/fss_tau.py, for visual consistency across every figure
# in the project.
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 18,

    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,

    "axes.linewidth": 1.0,

    "xtick.direction": "in",
    "ytick.direction": "in",

    "xtick.top": True,
    "ytick.right": True,

    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,

    "legend.fontsize": 18,
})


# One single plot per observable (not one panel per L): every L's tau(M)
# curve is overlaid on the same axes, colour-coded by L, with a dot
# marking the Sokal-selected window for each curve. Total output is
# exactly two figures -- tau_vs_lag_abs_m and tau_vs_lag_E -- regardless
# of how many L values the simulation has.
def make_overlay(obs_label, obs_tex, get_series):
    fig, ax = plt.subplots(figsize=(10, 7))

    curves = []
    global_M_max = 0

    for i in range(metadata.shape[0]):
        L = metadata["L"][i]
        beta_i = metadata["beta_i"][i]
        beta_f = metadata["beta_f"][i]
        n_beta = metadata["n_beta"][i]

        betas = np.array([beta_i]) if n_beta == 1 else np.linspace(beta_i, beta_f, n_beta)
        beta = BETA_OVERRIDE.get(L, betas[np.argmin(np.abs(betas - beta_c))])

        data = pd.read_csv(f"data/{sim_name}/{alg}_L{L}_beta{beta:.4f}.csv")
        x = get_series(data)
        N = len(x)
        taus, rho = tau_running(x)
        M_sel = sokal_window(taus, N)

        if M_sel is not None:
            tau_hat = max(taus[M_sel], 0.5)
            # Show a healthy margin past the selected window so the
            # plateau (or lack of one) is visible, without dragging in the
            # very long, purely noisy tail all the way out to N/2.
            M_max_local = min(N // 2, max(int(MAX_LOCAL_MULT * M_sel), 20))
            print(f"L={L:<4} beta={beta:.4f} obs={obs_label}: N={N}, "
                  f"M*={M_sel}, tau={tau_hat:.2f}, N/tau={N/tau_hat:.0f}")
        else:
            tau_hat = None
            M_max_local = N // 2
            print(f"L={L:<4} beta={beta:.4f} obs={obs_label}: N={N}, "
                  f"finestra non convergente (vedi warning sopra)")

        global_M_max = max(global_M_max, M_max_local)
        curves.append((L, N, taus, M_sel, tau_hat))

    # Shared x-axis across every L: each curve is drawn out to whichever
    # is smaller between the shared range and its own N/2 (beyond N/2 the
    # FFT autocovariance estimate is unreliable -- too few pairs at that
    # lag -- so a short-N curve simply ends earlier rather than being
    # extended past where it's trustworthy).

    colors = plt.cm.Blues(np.linspace(0,1,len(curves)))

    for color, (L, N, taus, M_sel, tau_hat) in zip(colors, curves):
        M_max_plot = min(N // 2, global_M_max)
        M = np.arange(M_max_plot)
        line, = ax.plot(M, taus[:M_max_plot], color=color, lw=1.6, label=fr"$L={L}$")
        if M_sel is not None and M_sel < M_max_plot:
            ax.plot([M_sel], [tau_hat], marker="o", linestyle="None", color="black", ms=5)

    if(alg == "metropolis") :      
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel("M")    
    ax.set_ylabel(rf"$\tau_{{\mathrm{{int}}}}^{{{obs_tex}}}$")
    ax.legend(loc="lower right", ncol=2 if metadata.shape[0] > 4 else 1, fontsize=16)

    fig.tight_layout()
    out = f"results/figure/{sim_name}_{alg}_tau_vs_lag_{obs_label}.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Salvato: {out}")


make_overlay("abs_m", "|m|", lambda data: data["m"].abs().to_numpy())
make_overlay("E", "E", lambda data: data["E_per_site"].to_numpy())

plt.show()
