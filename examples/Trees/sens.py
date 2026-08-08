#!/usr/bin/python3
from cmath import sin
import os
import numpy as np


from esys.escript import *
from esys.finley import  ReadMesh
from esys.escript.linearPDEs import LinearSinglePDE, SolverOptions
from esys.weipa import saveSilo
from esys.escript.pdetools import Locator

electrodes=np.genfromtxt('stations.csv',delimiter=',', dtype=[('id', '<i8'), ('x', '<f8'), ('y', '<f8'), ('z', '<f8')] )
X={}
Xenum = []
for i, x,y,z in electrodes:
    j = f"s{i:03d}"
    X[j] = np.array((x,y,z))
    Xenum.append(j)
print(f"{len(X)} electrodes read.")
NumElectrodesPerRing = len(X)

SETS = {}

iA, iB = 1, (NumElectrodesPerRing//2) +1
A, B=f"s{iA:03d}", f"s{iB:03d}"
print(f"Injections : {iA}, {iB} -> {A}, {B}")
iMiN = []
for i in range(NumElectrodesPerRing):
    iM = i+1
    iN = (i + NumElectrodesPerRing//2)%NumElectrodesPerRing +1
    if not (iM in [iA, iB] or iN in [iA, iB]):
        iMiN.append( (iN, iM) )
SETS["HalfDiag"] = iA, iB, iMiN
#............................................
iA, iB = 1, (NumElectrodesPerRing//2) +1
A, B=f"s{iA:03d}", f"s{iB:03d}"
print(f"Injections : {iA}, {iB} -> {A}, {B}")
iMiN = []
for i in range(NumElectrodesPerRing-1):
    iM = i+1
    iN = iM +1
    if not (iM in [iA, iB] or iN in [iA, iB]):
        iMiN.append( (iN, iM) )
SETS["HalfOffset"] = iA, iB, iMiN
#............................................
iA, iB = 1, (NumElectrodesPerRing//4) +1
A, B=f"s{iA:03d}", f"s{iB:03d}"
print(f"Injections : {iA}, {iB} -> {A}, {B}")
iMiN = []
for i in range(NumElectrodesPerRing):
    iM = i+1
    iN = (i + NumElectrodesPerRing//2)%NumElectrodesPerRing +1
    if not (iM in [iA, iB] or iN in [iA, iB]):
        iMiN.append( (iN, iM) )
SETS["QuarterDiag"] = iA, iB, iMiN

#............................................
iA, iB = 1, (NumElectrodesPerRing//4) +1
A, B=f"s{iA:03d}", f"s{iB:03d}"
print(f"Injections : {iA}, {iB} -> {A}, {B}")
iMiN = []
for i in range(NumElectrodesPerRing-1):
    iM = i+1
    iN = iM +1
    if not (iM in [iA, iB] or iN in [iA, iB]):

        iMiN.append( (iN, iM) )
SETS["QuarterOffset"] = iA, iB, iMiN



SIGMA_BG = 1

domain = ReadMesh("tree.fly")
print("Mesh read.")
nodelocators = Locator(Solution(domain), [X[S] for S in Xenum])

#============================================
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
print(f"{len(uS)} source potentials calculated.")


for tn  in SETS:
    DIR="out"+tn
    mkDir(DIR)
    iA, iB, iMiN = SETS[tn]
    A, B = f"s{iA:03d}", f"s{iB:03d}"
    uAB = uS[A]-uS[B]
    print(f"potential {A}-{B} calculated: {str(uAB)}.")
    uAB_at_stations = nodelocators(uAB)
    kk=0
    for iM, iN in iMiN:
        M, N = f"s{iM:03d}", f"s{iN:03d}"
        F_ABMN = uAB_at_stations[Xenum.index(M)] - uAB_at_stations[Xenum.index(N)]

        uMN = uS[M]-uS[N]
        s_ABMN = SIGMA_BG / F_ABMN * interpolate(inner(grad(uAB, Function(domain)), grad(uMN, Function(domain))),
                                                ReducedFunction(domain))
        print(A, B, M, N, F_ABMN, " -> ", s_ABMN)
        R= (0.1 / (abs(s_ABMN) * np.pi / 6) ) ** (1./3)
        #saveSilo(f"output/s_{M}_{N}", s=abs(s_ABMN), resolution=R)
        saveSilo(os.path.join(DIR, f"s_{kk:03d}"), s=abs(s_ABMN), resolution=R)
        kk+=1
    #uMN_at_stations = nodelocators(uMN)
    #print(uMN_at_stations)


1/0

iA, iB, iM, iN = tuple([self.schedule.getStationNumber(ST) for ST in stations])
F_ABMN = potential_at_stations[iA][iM] - potential_at_stations[iA][iN] - potential_at_stations[iB][iM] + \
         potential_at_stations[iB][iN]
F_ABMN_1 = (self.source_potential_at_station[iA][iM] - self.source_potential_at_station[iA][iN] -
            self.source_potential_at_station[iB][iM] + self.source_potential_at_station[iB][iN]) * self.sigma_src
UAB = potential[iA] - potential[iB]
UMN = potential[iM] - potential[iN]
sigma_a = F_ABMN_1 / F_ABMN
s_ABMN = sigma_a / F_ABMN * interpolate(inner(grad(UAB, Function(self.domain)), grad(UMN, Function(self.domain))),
                                        ReducedFunction(self.domain))
s_ABMN_max = Lsup(s_ABMN)
print(stations, " -> sigma_a = ", sigma_a, "; s = ", s_ABMN_max)


1/0

#=====================
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


