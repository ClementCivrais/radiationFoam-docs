#!/usr/bin/env python3
"""Line-shape figure of the theory page line_shapes.rst, computed from the formulas of
src/radiation/lineShapes/derived/voigtWhitingOliveroLongbothum (present, red) and compared with the Voigt
profile obtained by numerical convolution (scipy.special.voigt_profile, baseline, black).

(a) normalised profile wV*phi against |lambda - lambda0|/wV for three Lorentz fractions;
(b) Voigt full width at half maximum over (wL + wD): Liu et al. (2001) width of the atomic lines,
    wL/2 + sqrt(wL^2/4 + wD^2) of the diatomic bands, Olivero and Longbothum (1977), against the width of the
    numerical Voigt profile.
Run: python3 fig_line_shapes.py (writes line_shapes_voigt.png next to this script).
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import voigt_profile
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9,
                     "xtick.direction": "out", "ytick.direction": "out", "legend.frameon": False})
LN2 = np.log(2.0)


def width_liu(wL, wD):
    """Voigt FWHM of voigtWhitingOliveroLongbothum.C (Liu et al. 2001)"""
    d = (wL - wD) / (wL + wD)
    beta = 0.023665 * np.exp(0.6 * d) + 0.00418 * np.exp(-1.9 * d)
    r = 1.0 - 0.18121 * (1.0 - d**2) - beta * np.sin(np.pi * d)
    return r * (wL + wD)


def width_bands(wL, wD):
    """Voigt FWHM of diatomicBands.C"""
    return wL / 2.0 + np.sqrt(wL**2 / 4.0 + wD**2)


def width_ol(wL, wD):
    """Olivero and Longbothum (1977)"""
    return 0.5346 * wL + np.sqrt(0.2166 * wL**2 + wD**2)


def whiting(x, wL, wV):
    """Whiting (1968) shape as coded (wing term inside one exponential), x = |dlambda|/wV, [1/m] times wV"""
    a = wL / wV
    c0 = 1.0 / ((1.065 + (0.447 + 0.058 * a) * a) * wV)
    c2 = a * c0
    c1 = c0 - c2
    x225 = x**2.25
    return (c1 * np.exp(-2.772 * x**2) + c2 / (1.0 + 4.0 * x**2)
            + 0.016 * c2 * (1.0 - a) * np.exp(-0.4 * x225 - 10.0 / (10.0 + x225)))


def voigt(dl, wL, wD):
    """Voigt profile [1/m] of Lorentz FWHM wL and Gauss FWHM wD (numerical convolution)"""
    sigma = wD / (2.0 * np.sqrt(2.0 * LN2))
    return voigt_profile(dl, sigma, wL / 2.0)


def fwhm_numerical(wL, wD):
    p0 = voigt(0.0, wL, wD)
    return 2.0 * brentq(lambda x: voigt(x, wL, wD) - 0.5 * p0, 0.0, 10.0 * (wL + wD))


fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0))

# (a) profiles
ax = axs[0]
x = np.linspace(0.0, 25.0, 4000)
wD = 1.0
for wL, ls, lab in [(1e-6, "-", r"$w_L/w_D=0$"), (1.0, "--", r"$w_L/w_D=1$"), (20.0, ":", r"$w_L/w_D=20$")]:
    wV = width_liu(wL, wD)
    ax.semilogy(x, whiting(x, wL, wV) * wV, color="red", ls=ls, lw=1.2, label="present, " + lab)
    ax.semilogy(x, voigt(x * wV, wL, wD) * wV, color="black", ls=ls, lw=0.7)
ax.plot([], [], color="black", lw=0.7, label="Voigt, numerical")
ax.set_xlabel(r"$|\lambda-\lambda_0|/w_V$")
ax.set_ylabel(r"$w_V\,\phi$")
ax.set_ylim(1e-6, 2.0)
ax.set_xlim(0, 25)
ax.legend(fontsize=7, loc="upper right")
ax.set_title("(a) profile shape", fontsize=9)

# (b) widths
ax = axs[1]
ratio = np.logspace(-3, 3, 61)          # wL/wD
wD = 1.0
dvals = (ratio - 1.0) / (ratio + 1.0)
num = np.array([fwhm_numerical(r * wD, wD) for r in ratio]) / (ratio * wD + wD)
ax.plot(dvals, num, color="black", lw=1.0, label="Voigt, numerical")
ax.plot(dvals, width_liu(ratio * wD, wD) / (ratio * wD + wD), "o", color="red", mfc="red", ms=3.5,
        label="atomic lines (Liu et al. 2001)")
ax.plot(dvals, width_bands(ratio * wD, wD) / (ratio * wD + wD), "s", color="red", mfc="none", ms=3.5,
        label=r"diatomic bands ($w_L/2+\sqrt{w_L^2/4+w_D^2}$)")
ax.plot(dvals, width_ol(ratio * wD, wD) / (ratio * wD + wD), color="black", ls="--", lw=0.7,
        label="Olivero and Longbothum (1977)")
ax.set_xlabel(r"$d=(w_L-w_D)/(w_L+w_D)$")
ax.set_ylabel(r"$w_V/(w_L+w_D)$")
ax.legend(fontsize=6.5, loc="lower center")
ax.set_ylim(0.6, 1.08)
ax.set_title("(b) Voigt width", fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(HERE, "line_shapes_voigt.png"), dpi=200)

# Numbers quoted on the page
for r in [0.0, 1.0, 20.0]:
    wL = max(r, 1e-6)
    wV = width_liu(wL, 1.0)
    xs = np.linspace(0, 25, 200001)
    area = 2 * np.trapz(whiting(xs, wL, wV), xs * wV)
    vfrac = 2 * np.trapz(voigt(xs * wV, wL, 1.0), xs * wV)
    print("wL/wD=%g: wV=%.6f numerical FWHM=%.6f  Whiting area on +-25wV=%.6f  Voigt mass in +-25wV=%.6f"
          % (r, wV, fwhm_numerical(wL, 1.0), area, vfrac))
err = np.max(np.abs(width_liu(ratio, 1.0) - num * (ratio + 1)) / (num * (ratio + 1)))
errb = np.max(np.abs(width_bands(ratio, 1.0) - num * (ratio + 1)) / (num * (ratio + 1)))
erro = np.max(np.abs(width_ol(ratio, 1.0) - width_liu(ratio, 1.0)) / width_liu(ratio, 1.0))
print("max rel error Liu %.2e, bands %.2e, |OL-Liu| %.2e" % (err, errb, erro))
