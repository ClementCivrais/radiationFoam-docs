#!/usr/bin/env python3
"""Slab radiance of shockTube1D (src/radiation/transport/derived/shockTube1D/shockTube1D.C):
I = -(j/kappa) expm1(-kappa D), plotted as I/(j D) = -expm1(-tau)/tau against the optical thickness tau = kappa D
(present, red), with the optically thin limit I = j D and the optically thick limit I = j/kappa (black).
Run: python3 fig_slab.py (writes slab_radiance.png next to this script).
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9,
                     "xtick.direction": "out", "ytick.direction": "out", "legend.frameon": False})

tau = np.logspace(-4, 3, 400)
fig, ax = plt.subplots(figsize=(3.6, 3.0))
ax.loglog(tau, -np.expm1(-tau) / tau, color="red", lw=1.4, label=r"present, $-\mathrm{expm1}(-\tau)/\tau$")
ax.loglog(tau, np.ones_like(tau), color="black", ls="--", lw=0.8, label=r"thin limit, $I=jD$")
ax.loglog(tau, 1.0 / tau, color="black", ls=":", lw=0.8, label=r"thick limit, $I=j/\kappa$")
ax.set_xlabel(r"$\tau=\kappa_\lambda D$")
ax.set_ylabel(r"$I_\lambda/(j_\lambda D)$")
ax.set_ylim(5e-4, 2.0)
ax.legend(fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "slab_radiance.png"), dpi=200)

# Round-off of the naive form (1 - exp(-tau))/tau at small tau, quoted on the page
for t in [1e-8, 1e-12, 1e-16]:
    naive = (1.0 - np.exp(-t)) / t
    good = -np.expm1(-t) / t
    print("tau=%g naive=%.17g expm1=%.17g" % (t, naive, good))
