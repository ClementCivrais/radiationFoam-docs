#!/usr/bin/env python3
"""Two ingredients of diatomicBands (src/radiation/radiators/derived/diatomicBands).

(a) Hoenl-London factors of a type-2 band (perpendicular, Delta Lambda = 1), Pi -> Sigma
    (Lambda' = 1, Lambda'' = 0), the singlet forms of diatomicBandsLines.C, and their sum
    over the P, Q, R branches of one upper level, 2(2J'+1) (the code halves c_S for
    every type-2 band so that the lines of one level sum to 2J'+1).
(b) Term values F(J) - B J(J+1) of the two spin components of NO X 2Pi (v = 0, constants of
    src/radiation/database/molecules/NO.dat): closed form of the code (type 3/5 term,
    Hill and Van Vleck) as lines, eigenvalues of the Hund's case (a) Hamiltonian that
    the matrix method diagonalises as markers.

Run from this directory: python3 plot_hoenl_london_terms.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

# (a) type 2, Pi -> Sigma: lambdaUp false, L = -Lambda' = -1
L = -1.0
J = np.arange(1, 31, dtype=float)
SP = (J + 1 + L)*(J + 2 + L)/(J + 1)          # J'' = J' + 1
SQ = (J - L)*(2*J + 1)*(J + 1 + L)/(J*(J + 1))  # J'' = J'
SR = (J - L)*(J - 1 - L)/J                    # J'' = J' - 1
ax1.plot(J, SP, "k-", lw=1, label="P ($J''=J'+1$)")
ax1.plot(J, SQ, "k--", lw=1, label="Q ($J''=J'$)")
ax1.plot(J, SR, "k-.", lw=1, label="R ($J''=J'-1$)")
ax1.plot(J, SP + SQ + SR, "k:", lw=1.5, label="P+Q+R $= 2(2J'+1)$")
ax1.set_xlabel("$J'$")
ax1.set_ylabel("Hoenl-London factor $S_{J'J''}$")
ax1.set_title("(a) type 2, $^1\\Pi \\to {}^1\\Sigma$")
ax1.legend(frameon=False, fontsize=8)

# (b) NO X 2Pi, v = 0
Be, alphae, A = 1.70486, 0.017475, 123.26        # cm^-1, NO.dat [X2Pi]
B = Be - alphae*0.5
Y = A/B
Lam, S = 1.0, 0.5
Jh = np.arange(0.5, 40.5, 1.0)


def f3(j, sgn):
    return B*((j + 0.5)**2 - Lam**2 + sgn*0.5*np.sqrt(4*(j + 0.5)**2 - 4*Y*Lam**2 + (Y*Lam)**2))


def case_a(j):
    basis = [(-S + k, Lam - S + k) for k in range(2) if abs(Lam - S + k) <= j + 1e-9]
    x = j*(j + 1)
    n = len(basis)
    H = np.zeros((n, n))
    for a, (Sa, Oa) in enumerate(basis):
        H[a, a] = B*(x - Oa**2 + S*(S + 1) - Sa**2) + A*Lam*Sa
        for b, (Sb, Ob) in enumerate(basis):
            if abs(Sb - Sa - 1) < 1e-9 and abs(Ob - Oa - 1) < 1e-9:
                H[a, b] = H[b, a] = -B*np.sqrt(max(0, x - Oa*(Oa + 1)))*np.sqrt(max(0, S*(S + 1) - Sa*(Sa + 1)))
    return np.sort(np.linalg.eigvalsh(H))


ref = B*Jh*(Jh + 1)
ax2.plot(Jh, f3(Jh, -1) - ref, "k-", lw=1, label="closed form, $F_1$")
ax2.plot(Jh, f3(Jh, +1) - ref, "k--", lw=1, label="closed form, $F_2$")
for i, j in enumerate(Jh):
    w = case_a(j)
    lab = "case (a) matrix eigenvalues" if i == 0 else None
    if len(w) == 2:
        ax2.plot([j, j], w - B*j*(j + 1), "o", color="red", mfc="none", ms=4, label=lab)
    else:
        ax2.plot([j], w - B*j*(j + 1), "o", color="red", mfc="none", ms=4, label=lab)
ax2.set_xlabel("$J$")
ax2.set_ylabel("$F(J) - B_v J(J+1)$ [cm$^{-1}$]")
ax2.set_title("(b) NO X $^2\\Pi$, $v=0$, $Y=A/B_v=%.1f$" % Y)
ax2.legend(frameon=False, fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(HERE, "hoenl_london_terms.png"), dpi=150)
