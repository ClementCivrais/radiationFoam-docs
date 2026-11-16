#!/usr/bin/env python3
"""Figures of theory/instrument.rst, computed from the formulas of
src/radiation/instrument (no OpenFOAM run):

instrument_ils.png  the spectral kernels (ILS) of the EAST UV/Vis camera of
                    shot 50-29 (wg 0.28 nm, wl 0.070 nm, tutorials/EAST/
                    east5029): sqrt-Voigt ILS1 normalised on |s| <= 10 nm,
                    Gauss+Lorentz ILS2 (a = 0.2, illustrative), the Voigt
                    and the Gaussian of the same half widths;
instrument_srf.png  the spatial resolution function of the 50-29 VUV
                    camera: optics triangle (base 0.041 cm), camera
                    sqrt-Voigt (wg 0.026, wl 0.063 cm, cut at 10 cm), gate
                    top-hat (0.514 cm), and their composite.

All curves are analytical (black lines); the kernels are built as in the
C++ classes (exact Voigt by convolution, numerical normalisation on the
cut-off range).
"""
import os
import sys

import numpy as np
from scipy.special import voigt_profile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', '..', '..',
                                'tutorials', 'instrumentTests'))
from plot_style import plt, INK, SQUARE, save  # noqa: E402

LN2 = np.log(2.0)


def voigt_hw(s, wg, wl):
    """Voigt of Gaussian and Lorentzian HALF widths wg, wl (unit area)."""
    return voigt_profile(s, wg / np.sqrt(2 * LN2), wl)


def normalise(s, p):
    return p / np.trapz(p, s)


# ------------------------------------------------------------------ ILS
wg, wl, R, a = 0.28, 0.070, 10.0, 0.2          # nm
s = np.linspace(-R, R, 400001)
ils1 = normalise(s, np.sqrt(voigt_hw(s, wg, wl)))
G = np.sqrt(LN2 / np.pi) / wg * np.exp(-LN2 * (s / wg) ** 2)
L = 1 / (np.pi * wl) / ((s / wl) ** 2 + 1)
ils2 = normalise(s, (G + a * L) / (1 + a))
voigt = normalise(s, voigt_hw(s, wg, wl))

fig, ax = plt.subplots(figsize=SQUARE)
ax.plot(s, ils1, '-', color=INK, label='sqrt-Voigt (ILS1), $R$ = 10 nm')
ax.plot(s, ils2, '--', color=INK, label='Gauss + Lorentz (ILS2), $a$ = 0.2')
ax.plot(s, voigt, ':', color=INK, label='Voigt, same $w_G$, $w_L$')
ax.plot(s, G, '-.', color=INK, lw=0.8, label='Gaussian, same $w_G$')
ax.set_yscale('log')
ax.set_xlim(-4, 4)
ax.set_ylim(1e-4, 1e3)
ax.set_xlabel(r'$s = \lambda - \lambda_0$ (nm)')
ax.set_ylabel(r'$K(s)$ (nm$^{-1}$)')
ax.legend(loc='upper right')
save(fig, os.path.join(HERE, 'instrument_ils'))

# ------------------------------------------------------------------ SRF
h = 0.0005                                       # cm
x = np.arange(-12, 12 + h / 2, h)
b = 0.041
opt = np.clip(1 - 2 * np.abs(x) / b, 0, None) * 2 / b
cam = np.where(np.abs(x) <= 10, np.sqrt(voigt_hw(x, 0.026, 0.063)), 0)
cam = normalise(x, cam)
w = 0.514
mot = np.where(np.abs(x) < w / 2, 1 / w, 0.0)
srf = np.convolve(np.convolve(opt, cam, 'same') * h, mot, 'same') * h
srf = normalise(x, srf)

fig, ax = plt.subplots(figsize=SQUARE)
ax.plot(x, cam, '--', color=INK, label='camera (sqrt-Voigt)')
ax.plot(x, mot, ':', color=INK, label=r'gate, $U_s t_\mathrm{gate}$ = 0.514 cm')
ax.plot(x, srf, '-', color=INK, label='composite SRF')
ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-0.05, 3.0)
ax.set_xlabel(r'$s$ (cm)')
ax.set_ylabel(r'$K(s)$ (cm$^{-1}$)')
ax.legend(loc='upper right')
save(fig, os.path.join(HERE, 'instrument_srf'))
print('optics triangle peak', opt.max(), 'cm^-1 (off the axes: base 0.041 cm)')
