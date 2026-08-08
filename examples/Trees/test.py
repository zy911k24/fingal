#!/usr/bin/python3
from cmath import sin
from tkinter import Scale

from esys.escript import *
from esys.finley import  ReadMesh
from esys.escript.linearPDEs import LinearSinglePDE, SolverOptions
from esys.weipa import saveSilo

A, B="s001", "s025"
X = { A : (0.016350782307535765, 0.24946473080965087, 0.0),
      B : (-0.016350782307535717, -0.24946473080965087, 0.0) }

SIGMA_BG = 1

domain = ReadMesh("tree.fly")
print("Mesh read.")

# ....Background potential .....
pde = LinearSinglePDE(domain, isComplex=False)
pde.setValue(A = SIGMA_BG * kronecker(3) )
pde.setSymmetryOn()
n = domain.getNormal()
xBC = n.getX()
uS = {}
for S in X.keys():
    source = Scalar(0., DiracDeltaFunctions(domain))
    source.setTaggedValue(S, 1)
    r = xBC - X[S]
    alpha = SIGMA_BG * inner(r, n) / length(r) ** 2
    pde.setValue(d=alpha, y_dirac = source, y=Data(), X=Data())  # doi:10.1190/1.1440975
    uS[S] = pde.getSolution()
    print(S, str(uS[S]))
    pde.setValue(y=-alpha*uS[S], X=- SIGMA_BG*grad(uS[S]))  # doi:10.1190/1.1440975
    du=pde.getSolution()
    print(S, str(du))
    uS[S]+=du
u_iso = uS[A]-uS[B]

# lets try some anisotropy
X= Function(domain).getX()
angle = atan2(X[1], X[0])
# anisotopy factors
a_z = 1.
a_phi = 1.
###
sigma_phi = a_pi * SIGMA_BG
sigma_r   =        SIGMA_BG

A = pde.getCoefficients("A")
A[0,0] = sigma_phi * cos(angle)**2 + sigma_r * sin(angle)**2
A[0,1] = ( sigma_r - sigma_phi) * sin(angle) * cos(angle)
A[0,2] = 0
A[1,0] = A[0,1]
A[1,1] = sigma_phi * sin(angle)**2 + sigma_r * cos(angle)**2
A[1,2] = 0
A[2,0] = 0
A[2,1] = 0
A[2,2] = 1
pde.setValue(A = A )


print(angle)
saveSilo("result", angle = angle)


