#!/usr/bin/python3
#
# OBSOLETE - do not use.
#
# this builds the cylindrical (r,t,z) box (treegeocyl.tpl) with gmsh, but the t direction
# has to be periodic and ReadGmsh does not carry periodicity over, so the mesh it
# produces is unusable: the t=0 and t=2*Pi faces stay unconnected.
# the cylindrical formulation now lives in sensCWenner.py, which builds its own
# Brick(..., periodic1=True) and reads no mesh file at all.
# the cartesian route is mkMesh.py (treegeoxyz.tpl) + sensWenner.py.
#
# NOTE: this script writes tree.geo/tree.msh/tree.fly/stations.csv, i.e. the same
# names mkMesh.py uses. running it leaves sensWenner.py reading a cylindrical mesh
# as if it were cartesian.
#
import math
TreeDiameter = 0.5
MeshFF=0.8
MeshFF=1.
#TreeDiameter = 1.0
#MeshFF=0.8
#TreeDiameter = 2.0
#MeshFF=0.6

NumElectrodesPerRing = 48
NumberOfRings = 1
DistanceOfRings = 3.14 * TreeDiameter /NumElectrodesPerRing
RadIncrement = math.pi *2 /NumElectrodesPerRing

CoreThickness = TreeDiameter
Padding = TreeDiameter *5

MeshSizeCore = RadIncrement/5 *MeshFF
MeshSizePadding = RadIncrement * 2 *MeshFF
MeshSizeElectrodes = RadIncrement/20 * MeshFF


RadOffset = RadIncrement/2

TEMPLATE_FILE="./treegeocyl.tpl"

assert DistanceOfRings * (NumberOfRings+2) < CoreThickness

electrodes = {}
# point generation:
R = TreeDiameter/2
FirstRing = - DistanceOfRings * (NumberOfRings - 1) /2

GEOPOINTS =""
for i in range(NumberOfRings):
    z = (FirstRing + i * DistanceOfRings)
    for j in range(NumElectrodesPerRing):
        theta = j*RadIncrement+RadOffset
        pointid=i*100+j+1
        GEOPOINTS += f"Point({1000+pointid})={{TreeDiameter/2 , {theta}, {z}, MeshSizeElectrodes}};\n"
        GEOPOINTS += f"Point{{{1000+pointid}}} In Surface {{407}};\n"
        print(i, j, "id ", pointid, "->", math.degrees(theta))
        electrodes[pointid] = (TreeDiameter/2, theta ,z)
with open("stations.csv", 'w') as f:
    for station, X in electrodes.items():
        f.write(f"{station}, {X[0]}, {X[1]}, {X[2]}\n")

#print(GEOPOINTS)
geometry = open(TEMPLATE_FILE, 'r').read().format(TreeDiameter =  TreeDiameter,
                                                  CoreThickness = CoreThickness,
                                                  Padding = Padding,
                                                  MeshSizeCore = MeshSizeCore,
                                                  MeshSizePadding = MeshSizePadding,
                                                  MeshSizeElectrodes = MeshSizeElectrodes,
                                                  ELECTRODES = GEOPOINTS)
with open("tree.geo", 'w') as f:
    f.write(geometry)

if True:
    import subprocess
    rp=subprocess.run(["gmsh", "-3", "-optimize_netgen", "-o","tree.msh", "tree.geo"])
    #rp = subprocess.run(["gmsh", "-3", "-o", "tree.msh", "tree.geo"])
    dts = [f"s{s:03d}" for s in electrodes.keys()]
    dps = [electrodes[s] for s in electrodes.keys()]
    from esys.finley import ReadGmsh
    domain = ReadGmsh("tree.msh", 3, diracPoints=dps, diracTags=dts, optimize=True)
    domain.write("tree.fly")
    print("fly file created. Have a nice day!")
