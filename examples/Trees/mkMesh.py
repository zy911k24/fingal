#!/usr/bin/python3
import math
TreeDiameter = 0.5
MeshFF=1.0
#TreeDiameter = 1.0
#MeshFF=0.8
#TreeDiameter = 2.0
MeshFF=0.6

NumElectrodesPerRing = 48
NumberOfRings = 1
DistanceOfRings = 3.14 * TreeDiameter /NumElectrodesPerRing
RadIncrement = math.pi *2 /NumElectrodesPerRing

CoreThickness = TreeDiameter
Padding = TreeDiameter *5

MeshSizeCore = RadIncrement * TreeDiameter /4 *MeshFF
MeshSizePadding = RadIncrement * TreeDiameter *MeshFF
MeshSizeElectrodes = RadIncrement * TreeDiameter /10 * MeshFF

# refinement around the tree axis, so the pith regularisation radius R0 is resolved:
# AxisRefineRadius is the radius of the fine zone, MeshSizeAxis the size within it.
AxisRefineRadius = TreeDiameter/50
MeshSizeAxis = MeshSizeCore/5


RadOffset = RadIncrement/2

# cartesian (x,y,z) template: the electrode points below are cartesian and are
# placed "In Surface {203..206}", which only exist in the xyz template.
# treegeocyl.tpl is the cylindrical (r,t,z) box used by the obsolete mkMeshCyl.py.
TEMPLATE_FILE="./treegeoxyz.tpl"

assert DistanceOfRings * (NumberOfRings+2) < CoreThickness

electrodes = {}
# point generation:
R = TreeDiameter/2
FirstRing = - DistanceOfRings * (NumberOfRings - 1) /2
SEGMENTS = {1 : 205, 2 : 204, 3: 203, 4: 206}
GEOPOINTS =""
for i in range(NumberOfRings):
    z = (FirstRing + i * DistanceOfRings)
    for j in range(NumElectrodesPerRing):
        a = j*RadIncrement+RadOffset

        x,y  = math.sin(a), math.cos(a)
        pointid=i*100+j+1
        GEOPOINTS += f"Point({1000+pointid})={{{x} * TreeDiameter/2 , {y} * TreeDiameter/2, {z}, MeshSizeElectrodes}};\n"
        GEOPOINTS += f"Point{{{1000+pointid}}} In Surface {{{SEGMENTS[math.ceil(a/(math.pi/2))]}}};\n"
        print(i, j, math.degrees(j*RadIncrement+RadOffset), x,y,z, math.ceil(a/(math.pi/2)))
        electrodes[pointid] = (R*x,R*y,z)

#print(GEOPOINTS)
geometry = open(TEMPLATE_FILE, 'r').read().format(TreeDiameter =  TreeDiameter,
                                                  CoreThickness = CoreThickness,
                                                  Padding = Padding,
                                                  MeshSizeCore = MeshSizeCore,
                                                  MeshSizePadding = MeshSizePadding,
                                                  MeshSizeElectrodes = MeshSizeElectrodes,
                                                  MeshSizeAxis = MeshSizeAxis,
                                                  AxisRefineRadius = AxisRefineRadius,
                                                  ELECTRODES = GEOPOINTS)
with open("tree.geo", 'w') as f:
    f.write(geometry)

with open("stations.csv", 'w') as f:
    for station, X in electrodes.items():
        f.write(f"{station}, {X[0]}, {X[1]}, {X[2]}\n")


if True:
    import subprocess
    #rp=subprocess.run(["gmsh", "-3", "-optimize_netgen", "-o","tree.msh", "tree.geo"])
    rp = subprocess.run(["gmsh", "-3", "-o", "tree.msh", "tree.geo"])
    dts = [f"s{s:03d}" for s in electrodes.keys()]
    dps = [electrodes[s] for s in electrodes.keys()]
    from esys.finley import ReadGmsh
    domain = ReadGmsh("tree.msh", 3, diracPoints=dps, diracTags=dts, optimize=True)
    domain.write("tree.fly")
    print("fly file created. Have a nice day!")
