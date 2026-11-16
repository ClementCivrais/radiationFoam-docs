#!/usr/bin/env python3
"""Data-flow diagram of radiationFoam (guide/architecture.rst).

Boxes: the inputs, the run-time selected families of src/radiation and the
outputs; arrows: the order in which radiationFoam.C and the transport models
use them.  Writes dataflow.png next to this script.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))

INK = "#1a1a1a"
FILL_IN = "#f2f2f2"    # inputs and outputs
FILL_LIB = "#e3ecf5"   # library families
FILL_TR = "#fbe9e7"    # transport layer

fig, ax = plt.subplots(figsize=(9.0, 8.2))
ax.set_xlim(0, 100)
ax.set_ylim(0, 92)
ax.axis("off")


def box(x, y, w, h, title, sub, fill):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.3,rounding_size=1.0",
                                fc=fill, ec=INK, lw=0.9))
    ax.text(x + w / 2, y + h - 1.8, title, ha="center", va="top",
            fontsize=9.5, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 5.4, sub, ha="center", va="top",
            fontsize=7.8, color=INK, linespacing=1.3)


def arrow(x0, y0, x1, y1, label=None, side="right"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9))
    if label:
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        if side == "right":
            ax.text(xm + 1.2, ym, label, fontsize=7.4, style="italic",
                    ha="left", va="center", color=INK)
        else:
            ax.text(xm, ym + 1.3, label, fontsize=7.4, style="italic",
                    ha="center", va="bottom", color=INK)


# Row 1: inputs
box(1, 76, 30, 14, "src/radiation/database",
    "atoms/*.dat (levels, lines)\nmolecules/*.dat (band systems)\n*_photoionisation.dat", FILL_IN)
box(35, 76, 30, 14, "cell states",
    "cellState (singleCell)\nmacro file or dsmcFoam case\ndsmcRadiationDistribution files", FILL_IN)
box(69, 76, 30, 14, "system/radiationProperties",
    "selects every model by name:\nmodel, lineShape, populationModel,\ncontinuum, mode", FILL_IN)

# Row 2: library families
box(1, 48, 30, 19, "radiators/  +  lineShapes/",
    "radiatingSpecies (entry model):\natomicLines, diatomicBands\nlineShapeModel (entry lineShape):\nvoigtWhitingOliveroLongbothum", FILL_LIB)
box(35, 48, 30, 19, "populations/",
    "radiationPopulationModel\n(entry populationModel):\nboltzmannTemperatures\ncellDistribution", FILL_LIB)
box(69, 48, 30, 19, "continuum/",
    "continuumProcesses\n(entry continuum: none | physical):\nboundFreePhysical\nfreeFreePhysical", FILL_LIB)

# Row 3: transport
box(1, 25, 98, 15, "transport/   radiationTransportModel (entry mode)",
    "singleCell: j and kappa of one cell on spectralGrid (no transfer)\n"
    "shockTube1D: cell states from profiles/shockTubeProfile; every cell a slab of the tube diameter D,\n"
    "I = -(j/kappa) expm1(-kappa D) on every window; band integrals", FILL_TR)

# Row 4: instrument and outputs
box(1, 2, 44, 15, "instrument/   (shockTube1D, optional)",
    "shockTubeInstrument per camera:\nILS in wavelength, SRF in x,\nbands, shock front, spectra", FILL_LIB)
box(55, 2, 44, 15, "postProcessing/radiationFoam/",
    "spectrum.csv, populations_<s>.csv (singleCell)\nbands.csv, profile.csv, spectra_<w>.bin\n"
    "instrument_<camera>_bands.csv / _spectra.csv", FILL_IN)

# Arrows
arrow(16, 75.7, 16, 67.3, "line lists", "right")
arrow(50, 75.7, 50, 67.3, "n, T, n(n,v,J)", "right")
arrow(34.7, 57.5, 31.3, 57.5)
arrow(65.3, 57.5, 68.7, 57.5)
ax.text(33, 59.3, "n_i", fontsize=7.4, style="italic", ha="center", va="bottom", color=INK)
ax.text(67, 59.3, "n_i", fontsize=7.4, style="italic", ha="center", va="bottom", color=INK)
arrow(16, 47.7, 16, 40.3, "j, kappa per species", "right")
arrow(84, 47.7, 84, 40.3, "continuum j, kappa", "right")
arrow(23, 24.7, 23, 17.3, "I(x, lambda)", "right")
arrow(77, 24.7, 77, 17.3, "raw outputs", "right")
arrow(45.3, 9.5, 54.7, 9.5)

fig.savefig(os.path.join(HERE, "dataflow.png"), dpi=200, bbox_inches="tight")
print("wrote", os.path.join(HERE, "dataflow.png"))
