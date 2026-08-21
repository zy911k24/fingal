#!/usr/bin/python3
import os

import numpy as np


from esys.escript import *
from esys.finley import  ReadMesh
from esys.escript.linearPDEs import LinearSinglePDE, SolverOptions
from esys.weipa import saveSilo
from esys.escript.pdetools import Locator

# -----------------
SIGMA_Z = 1e-3
SIGMA_R = 5e-4
SIGMA_T = 2e-4
#SIGMA_R, SIGMA_Z = SIGMA_T, SIGMA_T
# pith radius. the fibre direction e_r is undefined at r=0, so the anisotropy is
# blended to isotropic (SIGMA_T) inside R0. without this the conductivity tensor is
# discontinuous on the axis and the solution has a r**sqrt(SIGMA_T/SIGMA_R) cusp there
# that no mesh can resolve. R0 needs a few elements across it: see AxisRefineRadius
# and MeshSizeAxis in mkMesh.py, which refine the mesh along the axis for this.
R0 = 0.01
DIR = "output"
SILO = "resultAnisoSmallXYZ_R0"
# -----------------
mkDir(DIR)
def makeWennerArrayOnRing(numElectrodes=48, id0=1, amax=None):
    """
    creates a schedule (A, B, M, N) for a Wenner array on a closed ring of numElectrodes
    equally spaced electrodes with ids id0, ... , id0+numElectrodes-1. In contrast to the
    Wenner array on a line the array wraps around the ring.

    a is the dipole spacing counted in electrodes: A, M, N, B are at positions
    k, k+a, k+2*a, k+3*a (modulo numElectrodes) for each starting electrode k.
    3*a < numElectrodes is required so A and B remain distinct, which gives
    amax = (numElectrodes-1)//3 as largest u_geoable spacing.
    """
    if amax is None:
        amax = (numElectrodes - 1) // 3
    assert 3 * amax < numElectrodes, "spacing amax is too large for the number of electrodes."
    schedule = []
    for a in range(1, amax + 1):
        for k in range(numElectrodes):
            schedule.append((k + id0,
                             (k + 3 * a) % numElectrodes + id0,
                             (k + 1 * a) % numElectrodes + id0,
                             (k + 2 * a) % numElectrodes + id0))
    return schedule

electrodes=np.genfromtxt('stations.csv',delimiter=',', dtype=[('id', '<i8'), ('x', '<f8'), ('y', '<f8'), ('z', '<f8')] )
X={}
Xenum = []
for i, x,y,z in electrodes:
    j = f"s{i:03d}"
    X[j] = np.array((x,y,z))
    Xenum.append(j)
NumElectrodesPerRing = len(X)
print(f"{len(X)} electrodes read.")

# this is the cartesian (x,y,z) formulation: stations.csv and tree.fly must come from
# mkMesh.py. guard against a stale cylindrical (r,t,z) pair, which would otherwise be
# read as x-y without complaint and quietly produce nonsense.
radius = np.hypot(*np.array([X[s][:2] for s in Xenum]).T)
assert np.ptp(radius) < 0.01 * radius.mean(), (
    f"stations.csv is not a ring in the x-y plane (radius {radius.min():g}..{radius.max():g}) "
    "- these look like cylindrical (r,t,z) coordinates. rerun mkMesh.py.")
print(f"station ring radius = {radius.mean():g}")

schedule = makeWennerArrayOnRing(NumElectrodesPerRing)
print("Wenner schedule created.")

# ====================================

domain = ReadMesh("tree.fly")
print("mesh loaded.")

nodelocators = Locator(Solution(domain), [X[S] for S in Xenum])
#============================================
# ....geometry potential .....
pde = LinearSinglePDE(domain, isComplex=False)
pde.getSolverOptions().setTolerance(1e-10)
z = domain.getX()[2]
pde.setValue(A = kronecker(3) , q= whereZero(z-inf(z)) +  whereZero(z-sup(z)) )
pde.setSymmetryOn()
u_geo = {}
for S in X.keys():
    source = Scalar(0., DiracDeltaFunctions(domain))
    source.setTaggedValue(S, 1)
    pde.setValue(y_dirac = source)
    u_geo[S] = pde.getSolution()
    print(S, str(u_geo[S]))
print(f"{len(u_geo)} geometry potentials calculated.")
#============================================
# .... potentials  .....;
print("Now anisotropic potentials")
print(f"SIGMA_R = {SIGMA_R}")
print(f"SIGMA_T = {SIGMA_T}")
print(f"SIGMA_Z = {SIGMA_Z}")
A=pde.getCoefficient('A')
x= A.getX()
# softened radial direction: |e| = r/sqrt(r**2+R0**2), so |e| -> 0 on the axis and the
# in-plane tensor SIGMA_T*I + (SIGMA_R-SIGMA_T)*e><e becomes isotropic there instead of
# discontinuous. for r >> R0 this is the same unit vector as before.
e = x[:2]/sqrt(x[0]**2 + x[1]**2 + R0**2)
A[0,0] = SIGMA_T + ( SIGMA_R - SIGMA_T) * e[0]**2
A[0,1] = ( SIGMA_R - SIGMA_T) * e[1] * e[0]
A[0,2] = 0
A[1,0] = A[0,1]
A[1,1] = SIGMA_T + ( SIGMA_R - SIGMA_T) * e[1]**2
A[1,2] = 0
A[2,0] = 0
A[2,1] = 0
A[2,2] = SIGMA_Z
#x= ReducedFunction(domain).getX()
#theta = interpolate(atan2(x[1], x[0]), A.getFunctionSpace())

#A[0,0] = SIGMA_R * cos(theta)**2 + SIGMA_T * sin(theta)**2
#A[0,1] = ( SIGMA_R - SIGMA_T) * sin(theta) * cos(theta)
#A[0,2] = 0
#A[1,0] = A[0,1]
#A[1,1] = SIGMA_R * sin(theta)**2 + SIGMA_T * cos(theta)**2
#A[1,2] = 0
#A[2,0] = 0
#A[2,1] = 0
#A[2,2] = SIGMA_Z

pde.setValue(A = A )
u = {}
for S in X.keys():
    source = Scalar(0., DiracDeltaFunctions(domain))
    source.setTaggedValue(S, 1)
    pde.setValue(y_dirac = source)
    u[S] = pde.getSolution()
    print(S, str(u[S]))
print(f"{len(u)} anisotropic potentials calculated.")

s_t= Scalar(0, ReducedFunction(domain))
s_r= Scalar(0, ReducedFunction(domain))
s_z= Scalar(0, ReducedFunction(domain))

x= s_t.getX()
e = x[:2]/sqrt(x[0]**2 + x[1]**2 + R0**2)

step = 0
cc = 0
for iA, iB, iM, iN in schedule:
    A, B = f"s{iA:03d}", f"s{iB:03d}"
    M, N = f"s{iM:03d}", f"s{iN:03d}"
    step_new = NumElectrodesPerRing - (iA - iB) % NumElectrodesPerRing
    #if not step_new == step:
    #    if step> 0:
    #        saveSilo(os.path.join(DIR, f"s_{step:03d}"), s_r=s_r, s_t=s_t, s_z=s_z)
    #    step = step_new
    #print(A, B, M, N, NumElectrodesPerRing-(iA-iB)%NumElectrodesPerRing)
    u_geo_at_stations = nodelocators( u_geo[A] - u_geo[B])
    f_geo =  u_geo_at_stations[Xenum.index(M)] - u_geo_at_stations[Xenum.index(N)]

    uAB= u[A]-u[B]
    uMN= u[M]-u[N]
    uAB_at_stations = nodelocators(uAB)
    F_ABMN = uAB_at_stations[Xenum.index(M)] - uAB_at_stations[Xenum.index(N)]
    sigma_ABMN = f_geo/F_ABMN
    g_AB = grad(uAB, e.getFunctionSpace())
    g_MN = grad(uMN, e.getFunctionSpace())
    # dA/d_sigma_r = e><e and dA/d_sigma_t = I - e><e for the softened e, so the
    # tangential kernel is the in-plane product minus the radial one. (the previous
    # form (e0*g1-e1*g0) is the tangential component only while |e| = 1.)
    er_AB = e[0] * g_AB[0] + e[1] * g_AB[1]
    er_MN = e[0] * g_MN[0] + e[1] * g_MN[1]
    s_ABMN_r = sigma_ABMN / F_ABMN * er_AB * er_MN
    s_ABMN_t = sigma_ABMN / F_ABMN * ( g_AB[0] * g_MN[0] + g_AB[1] * g_MN[1] - er_AB * er_MN)
    s_ABMN_z = sigma_ABMN / F_ABMN * g_AB[2] * g_MN[2]
    print(A, B, M, N, "geo =", f_geo, "sigma_a =", sigma_ABMN)
    print("\t\t","s_ABMN_r max = ", Lsup(s_ABMN_r), "s_ABMN_t max = ", Lsup(s_ABMN_t),"s_ABMN_z max = ", Lsup(s_ABMN_z))
    s_r+=abs(s_ABMN_r)
    s_t+=abs(s_ABMN_t)
    s_z+=abs(s_ABMN_z)
    cc+=1

#saveSilo(os.path.join(DIR, f"s_{step:03d}"), s_r=s_r, s_t=s_t, s_z=s_z)
saveSilo(SILO, s_r=s_r, s_t=s_t, s_z=s_z)
print("sensity written to "+SILO)