#!/usr/bin/python3
import math


import numpy as np


from esys.escript import *
from esys.finley import  Brick
from esys.escript.linearPDEs import LinearSinglePDE, SolverOptions
from esys.weipa import saveSilo
from esys.escript.pdetools import Locator

TreeDiameter = 0.5

NumElectrodesPerRing = 48
NumberOfRings = 1
DistanceOfRings = 3.14 * TreeDiameter /NumElectrodesPerRing
RadIncrement = math.pi *2 /NumElectrodesPerRing

# -----------------
SIGMA_Z = 1e-3
SIGMA_R = 5e-4
SIGMA_T = 2e-4
#SIGMA_R, SIGMA_Z = SIGMA_T, SIGMA_T

# the electrode ring sits on the bottom face z=0, which carries no Dirichlet condition
# and is therefore the mirror plane of the physical problem: the domain solved here is
# the half z>0 of a medium symmetric about z=0. only half of the current injected at an
# electrode flows into this half, so a unit physical current is a source of 1/2:
# with the even extension of the test function, int_full = 2*int_{z>0}, hence
# 2*int_{z>0} sigma grad(u).grad(v) = I*v(x_A).
SOURCE_STRENGTH = 0.5

SILO = "resultAnisoSmallCyl"
#============================
CoreThickness = TreeDiameter
Padding = CoreThickness * 2
FirstRing = Padding + CoreThickness/2 - DistanceOfRings * (NumberOfRings - 1) /2
FirstRing = 0

#assert DistanceOfRings * (NumberOfRings+2) < CoreThickness
# -----------------
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

L_r=TreeDiameter/2
L_t=2*np.pi
L_z=CoreThickness/2 + Padding


ElementsBetweenElectrodes=8
h_r = math.pi *2 /(NumElectrodesPerRing * ElementsBetweenElectrodes)

NE_r=math.ceil(L_r/h_r)
NE_t=math.ceil(2*np.pi/h_r)
NE_z=math.ceil(L_z/h_r)
h=min(L_r/NE_r, L_t/NE_t, L_z/NE_z)
print(h)
print(f"domain extend = {L_r} x {L_t} x {L_z}")
print(f"expected number of elements in tangential direction = {NumElectrodesPerRing * ElementsBetweenElectrodes}")
print(f"grid = {NE_r} x {NE_t} x {NE_z} = {NE_r*NE_t*NE_z}")
print(f"spacing = {L_r/NE_r} x {L_t/NE_t} x {L_z/NE_z}")

X={}
Xenum = []
for i in range(NumberOfRings):
    for j in range(NumElectrodesPerRing):
        theta = j*RadIncrement
        k = f"s{i*100+j+1:03d}"
        X[k] = np.array((L_r, theta, FirstRing + i * DistanceOfRings) )
        Xenum.append(k)
NumElectrodesPerRing = len(X)
print(f"{len(X)} electrodes created.")

dts = [s  for s in X.keys()]
dps = [X[s] for s in X.keys()]
domain = Brick(NE_r, NE_t, NE_z, l0=L_r, l1=L_t, l2=L_z, diracPoints=dps, diracTags=dts, periodic1=True)

schedule = makeWennerArrayOnRing(NumElectrodesPerRing)
print("Wenner schedule created.")

#domain.write("test.fly")

nodelocators = Locator(DiracDeltaFunctions(domain), [X[S] for S in Xenum])
print(nodelocators.getX())

#============================================
# ....geometry potential .....
pde = LinearSinglePDE(domain, isComplex=False)
pde.getSolverOptions().setTolerance(1e-10)
pde.setSymmetryOn()
A=pde.createCoefficient('A')
r = A.getX()[0]
# grad_x(u) = (du/dr, 1/r du/dt, du/dz) and dxdydz = r dr dt dz, so the coefficient
# in the (r,t,z) computational box is diag(sigma_r*r, sigma_t/r, sigma_z*r).
# the 1/r couples the nodes on the axis: without it they stay independent
# degrees of freedom and u becomes multivalued at r=0.
r_safe = clip(r, minval=1e-12) # r=0 is not a quadrature point, this is defensive only
A[0,0] = r
A[0,1] = 0
A[0,2] = 0
A[1,0] = 0
A[1,1] = 1/r_safe
A[1,2] = 0
A[2,0] = 0
A[2,1] = 0
A[2,2] = r
z=domain.getX()[2]
pde.setValue(A = A , q= 0*whereZero(z-inf(z)) +  whereZero(z-sup(z)) )

u_geo = {}
for S in X.keys():
    source = Scalar(0., DiracDeltaFunctions(domain))
    source.setTaggedValue(S, SOURCE_STRENGTH)
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
r=A.getX()[0]
r_safe = clip(r, minval=1e-12)
A[0,0] = SIGMA_R *r
A[0,1] = 0
A[0,2] = 0
A[1,0] = 0
A[1,1] = SIGMA_T/r_safe
A[1,2] = 0
A[2,0] = 0
A[2,1] = 0
A[2,2] = SIGMA_Z * r
pde.setValue(A = A )
u = {}
for S in X.keys():
    source = Scalar(0., DiracDeltaFunctions(domain))
    source.setTaggedValue(S, SOURCE_STRENGTH)
    pde.setValue(y_dirac = source)
    u[S] = pde.getSolution()
    print(S, str(u[S]))
print(f"{len(u)} anisotropic potentials calculated.")

s_t= Scalar(0, ReducedFunction(domain))
s_r= Scalar(0, ReducedFunction(domain))
s_z= Scalar(0, ReducedFunction(domain))
r=A.getX()[0]
r_safe = clip(r, minval=1e-12)
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
    #print(A, B, M, N, NumElectrodesPerRing-(iA-iB)%NumElectrodesPerRing)\
    u_geo_at_stations = nodelocators( u_geo[A] - u_geo[B])
    f_geo =  u_geo_at_stations[Xenum.index(M)] - u_geo_at_stations[Xenum.index(N)]

    uAB= u[A]-u[B]
    uMN= u[M]-u[N]
    uAB_at_stations = nodelocators(uAB)
    F_ABMN = uAB_at_stations[Xenum.index(M)] - uAB_at_stations[Xenum.index(N)]
    sigma_ABMN = f_geo/F_ABMN
    g_AB = grad(uAB, r.getFunctionSpace())
    g_MN = grad(uMN, r.getFunctionSpace())
    # d/d_sigma of the bilinear form, as densities with respect to dr dt dz:
    # d/d_sigma_r -> r * du/dr dv/dr, d/d_sigma_t -> 1/r * du/dt dv/dt, d/d_sigma_z -> r * du/dz dv/dz
    s_ABMN_r = sigma_ABMN / F_ABMN  * r * g_AB[0] * g_MN[0]
    s_ABMN_t = sigma_ABMN / F_ABMN  / r_safe * g_AB[1] * g_MN[1]
    s_ABMN_z = sigma_ABMN / F_ABMN * r * g_AB[2] * g_MN[2]
    print(A, B, M, N, "geo =", f_geo, "sigma_a =", sigma_ABMN)
    print("\t\t","s_ABMN_r max = ", Lsup(s_ABMN_r), "s_ABMN_t max = ", Lsup(s_ABMN_t),"s_ABMN_z max = ", Lsup(s_ABMN_z))
    s_r+=abs(s_ABMN_r)
    s_t+=abs(s_ABMN_t)
    s_z+=abs(s_ABMN_z)
    cc+=1

#saveSilo(os.path.join(DIR, f"s_{step:03d}"), s_r=s_r, s_t=s_t, s_z=s_z)
# so far the kernels are densities with respect to dr dt dz. the mesh is remapped to
# cartesian (x,y,z) below, so divide out the jacobian r to get densities per physical
# volume, which is what sensWenner.py computes and what an inversion cell weights.
# this has to happen before setX, while getX() still returns (r,t,z).
inv_r = 1/clip(ReducedFunction(domain).getX()[0], minval=1e-12)
s_r *= inv_r
s_t *= inv_r
s_z *= inv_r

X= domain.getX()
v=Vector(0, X.getFunctionSpace())
v[0]=X[0]*cos(X[1])
v[1]=X[0]*sin(X[1])
v[2]=X[2]
domain.setX(v)
saveSilo(SILO, s_r=s_r, s_t=s_t, s_z=s_z)
print("sensity written to "+SILO)