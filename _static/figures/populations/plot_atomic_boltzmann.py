#!/usr/bin/env python3
"""Boltzmann populations of N I and O I as boltzmannTemperatures forms them
(src/radiation/populations/derived/boltzmannTemperatures), computed from the
level lists of src/radiation/database/atoms/{N_I,O_I}.dat.

(a) population per state n_i/(n g_i) of the N I levels at T_el = 10000 K,
    levels kept below the ionisation limit E_ion (n_e = 0, no lowering);
(b) share of the atoms that the levels at or above E_ion would hold if they
    were not cut, sum_{E_i >= E_ion} g_i exp(-E_i/kT) / Q_all, against T_el.

Run from this directory: python3 plot_atomic_boltzmann.py
(needs RADIATIONFOAM_DIR or the repository layout docs/OnlineDocumentation/...).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("RADIATIONFOAM_DIR",
                      os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "..")))
ATOMS = os.path.join(ROOT, "src", "radiation", "database", "atoms")

KB = 1.380649e-23
HC100 = 6.62607015e-34*299792458.0*100.0     # J per cm^-1
EV = 1.602176634e-19

# Ionisation energies [cm^-1] of the tutorials (NIST ASD 5.12)
EION = {"N_I": 117225.7, "O_I": 109837.02}


def read_levels(name):
    toks = []
    for ln in open(os.path.join(ATOMS, name + ".dat")):
        toks += ln.split("//")[0].split()
    n = int(toks[0])
    g = np.array([float(toks[1 + 5*i]) for i in range(n)])
    E = np.array([float(toks[2 + 5*i]) for i in range(n)])*HC100
    return g, E


def kept(E, Eb):
    """Levels 0..N_b-1, N_b = last level (file order) with E < E_b, plus one."""
    idx = np.nonzero(E < Eb)[0]
    nk = idx[-1] + 1 if idx.size else 0
    return np.arange(nk)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

# (a) N I at 10000 K
g, E = read_levels("N_I")
Eb = EION["N_I"]*HC100
k = kept(E, Eb)
T = 10000.0
Q = np.sum(g[k]*np.exp(-E[k]/(KB*T)))
pop = np.exp(-E[k]/(KB*T))/Q
ax1.semilogy(E[k]/EV, pop, "o", mfc="none", mec="k", ms=4, label="levels of N_I.dat, $E_i < E_\\mathrm{ion}$")
Eg = np.linspace(0, Eb/EV, 200)
ax1.semilogy(Eg, np.exp(-Eg*EV/(KB*T))/Q, "k-", lw=1, label="$\\exp(-E/kT_\\mathrm{el})/Q$")
ax1.axvline(Eb/EV, color="k", ls="--", lw=0.8)
ax1.text(Eb/EV - 0.2, 1e-2, "$E_\\mathrm{ion}$", ha="right")
ax1.set_xlabel("level energy $E_i$ [eV]")
ax1.set_ylabel("population per state $n_i/(n\\,g_i)$ [-]")
ax1.set_title("(a) N I, $T_\\mathrm{el}$ = 10000 K, $n_e$ = 0")
ax1.legend(frameon=False, fontsize=8, loc="lower left")

# (b) share of the autoionising levels if they were kept
Ts = np.linspace(5000, 30000, 101)
for name, ls in (("N_I", "-"), ("O_I", "--")):
    g, E = read_levels(name)
    Eb = EION[name]*HC100
    above = E >= Eb
    share = []
    for T in Ts:
        w = g*np.exp(-E/(KB*T))
        share.append(w[above].sum()/w.sum())
    ax2.plot(Ts/1000, 100*np.array(share), "k" + ls, lw=1.2,
             label=name.replace("_", " ") + " (%d of %d levels above $E_\\mathrm{ion}$)" % (above.sum(), len(E)))
ax2.set_xlabel("$T_\\mathrm{el}$ [kK]")
ax2.set_ylabel("share of the atoms above $E_\\mathrm{ion}$ if not cut [%]")
ax2.set_title("(b) weight removed by the level cut")
ax2.legend(frameon=False, fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(HERE, "atomic_boltzmann.png"), dpi=150)
for T in (20000.0, 25000.0):
    g, E = read_levels("O_I")
    w = g*np.exp(-E/(KB*T))
    print("O I share above E_ion at %g K: %.2f %%" % (T, 100*w[E >= EION["O_I"]*HC100].sum()/w.sum()))
