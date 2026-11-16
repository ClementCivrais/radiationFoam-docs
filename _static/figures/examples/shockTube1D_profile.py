#!/usr/bin/env python3
"""Input profile of tutorials/shockTube1D (macroFile/constant/profile.csv):
number densities and temperatures along the tube.  Plots the tutorial input
only (no OpenFOAM run).  Writes shockTube1D_profile.png."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..', '..', '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'tutorials', 'instrumentTests'))
from plot_style import plt, INK, SQUARE, save  # noqa: E402

d = np.genfromtxt(os.path.join(ROOT, 'tutorials', 'shockTube1D', 'macroFile', 'constant', 'profile.csv'),
                  delimiter=',', names=True, skip_header=1)
x = d['x'] * 1e3
fig, (a1, a2) = plt.subplots(2, 1, figsize=(SQUARE[0], 1.4 * SQUARE[1]), sharex=True)
for s, ls in (('N2', '-'), ('N', '--'), ('O', ':')):
    a1.plot(x, d['n_' + s], ls, color=INK, label=s.replace('N2', r'N$_2$'))
a1.set_yscale('log')
a1.set_ylim(1e14, 1e23)
a1.set_ylabel(r'$n$ (m$^{-3}$)')
a1.legend(loc='lower right')
a2.plot(x, d['Ttrans_N2'], '-', color=INK, label=r'columns Ttrans_<s>, Trot_N2')
a2.plot(x, d['Tvib_N2'], '--', color=INK, label=r'columns Tvib_N2, Telec_<s>')
a2.set_ylabel('Temperature (K)')
a2.set_xlabel(r'$x$ (mm)')
a2.legend(loc='upper right')
save(fig, os.path.join(HERE, 'shockTube1D_profile'))
