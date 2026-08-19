Mesh.MshFileVersion = 2.2;

TreeDiameter = {TreeDiameter};
CoreThickness = {CoreThickness};
Padding = {Padding};

MeshSizeCore = {MeshSizeCore};
MeshSizePadding = {MeshSizePadding};
MeshSizeElectrodes = {MeshSizeElectrodes};


// padding south
Point(1) = {{0, -0, -CoreThickness/2 - Padding, MeshSizePadding}};
Point(3) = {{-TreeDiameter/2, 0, -CoreThickness/2 - Padding, MeshSizePadding}};
Point(2) = {{0, -TreeDiameter/2,  -CoreThickness/2 - Padding, MeshSizePadding}};
Point(4) = {{TreeDiameter/2, 0, -CoreThickness/2 - Padding, MeshSizePadding}};
Point(5) = {{0, TreeDiameter/2, -CoreThickness/2 - Padding , MeshSizePadding}};
Circle(1) = {{4, 1, 5}};
Circle(2) = {{5, 1, 3}};
Circle(3) = {{3, 1, 2}};
Circle(4) = {{2, 1, 4}};
Line(5) = {{1, 4}};
Line(6) = {{1, 3}};
Curve Loop(1) = {{5, 1, 2, -6}};
Plane Surface(1) = {{-1}};
Curve Loop(2) = {{5, -4, -3, -6}};
Plane Surface(2) = {{2}};


// .... core
Point(101) = {{0, -0, -CoreThickness/2, MeshSizeCore}};
Point(103) = {{-TreeDiameter/2, 0, -CoreThickness/2, MeshSizeCore}};
Point(102) = {{0, -TreeDiameter/2,  -CoreThickness/2, MeshSizeCore}};
Point(104) = {{TreeDiameter/2, 0, -CoreThickness/2, MeshSizeCore}};
Point(105) = {{0, TreeDiameter/2, -CoreThickness/2, MeshSizeCore}};
Circle(101) = {{104, 101, 105}};
Circle(102) = {{105, 101, 103}};
Circle(103) = {{103, 101, 102}};
Circle(104) = {{102, 101, 104}};
Line(105) = {{101, 104}};
Line(106) = {{101, 103}};
Curve Loop(101) = {{105, 101, 102, -106}};
Plane Surface(101) = {{101}};
Curve Loop(102) = {{105, -104, -103, -106}};
Plane Surface(102) = {{-102}};
Line(107) = {{1, 101}};
Line(108) = {{3, 103}};
Line(109) = {{4, 104}};
Line(110) = {{2, 102}};
Line(111) = {{5, 105}};
Curve Loop(103) = {{3, 110, -103, -108}};
Surface(103) = {{103}};
Curve Loop(104) = {{4, 109, -104, -110}};
Surface(104) = {{104}};
Curve Loop(105) = {{1, 111, -101, -109}};
Surface(105) = {{105}};
Curve Loop(106) = {{2, 108, -102, -111}};
Surface(106) = {{106}};
Curve Loop(107) = {{5, 109, -105, -107}};
Plane Surface(107) = {{107}};
Curve Loop(108) = {{6, 108, -106, -107}};
Plane Surface(108) = {{108}};

Surface Loop(101) = {{2, 103, 104, 107, 108, 102}};
Volume(101) = {{101}};
Surface Loop(102) = {{107, 108, 1, 106, 105, 101}};
Volume(102) = {{102}};


Point(201) = {{0, -0,  CoreThickness/2, MeshSizeCore}};
Point(203) = {{-TreeDiameter/2, 0, CoreThickness/2, MeshSizeCore}};
Point(202) = {{0, -TreeDiameter/2,  CoreThickness/2, MeshSizeCore}};
Point(204) = {{TreeDiameter/2, 0, CoreThickness/2, MeshSizeCore}};
Point(205) = {{0, TreeDiameter/2, CoreThickness/2, MeshSizeCore}};
Circle(201) = {{204, 201, 205}};
Circle(202) = {{205, 201, 203}};
Circle(203) = {{203, 201, 202}};
Circle(204) = {{202, 201, 204}};
Line(205) = {{201, 204}};
Line(206) = {{201, 203}};
Curve Loop(201) = {{205, 201, 202, -206}};
Plane Surface(201) = {{201}};
Curve Loop(202) = {{205, -204, -203, -206}};
Plane Surface(202) = {{-202}};
Line(207) = {{101, 201}};
Line(208) = {{103, 203}};
Line(209) = {{104, 204}};
Line(210) = {{102, 202}};
Line(211) = {{105, 205}};
Curve Loop(203) = {{103, 210, -203, -208}};
Surface(203) = {{203}};
Curve Loop(204) = {{104, 209, -204, -210}};
Surface(204) = {{204}};
Curve Loop(205) = {{101, 211, -201, -209}};
Surface(205) = {{205}};
Curve Loop(206) = {{102, 208, -202, -211}};
Surface(206) = {{206}};
Curve Loop(207) = {{105, 209, -205, -207}};
Plane Surface(207) = {{207}};
Curve Loop(208) = {{106, 208, -206, -207}};
Plane Surface(208) = {{208}};

Surface Loop(201) = {{102, 203, 204, 207, 208, 202}};
Volume(201) = {{201}};
Surface Loop(202) = {{207, 208, 101, 206, 205, 201}};
Volume(202) = {{202}};
//--------------------------------
// padding north

Point(301) = {{0, -0,  CoreThickness/2 + Padding, MeshSizePadding}};
Point(303) = {{-TreeDiameter/2, 0, CoreThickness/2 + Padding, MeshSizePadding}};
Point(302) = {{0, -TreeDiameter/2,  CoreThickness/2 + Padding, MeshSizePadding}};
Point(304) = {{TreeDiameter/2, 0, CoreThickness/2 + Padding, MeshSizePadding}};
Point(305) = {{0, TreeDiameter/2, CoreThickness/2 + Padding, MeshSizePadding}};
Circle(301) = {{304, 301, 305}};
Circle(302) = {{305, 301, 303}};
Circle(303) = {{303, 301, 302}};
Circle(304) = {{302, 301, 304}};
Line(305) = {{301, 304}};
Line(306) = {{301, 303}};
Curve Loop(301) = {{305, 301, 302, -306}};
Plane Surface(301) = {{301}};
Curve Loop(302) = {{305, -304, -303, -306}};
Plane Surface(302) = {{-302}};
Line(307) = {{201, 301}};
Line(308) = {{203, 303}};
Line(309) = {{204, 304}};
Line(310) = {{202, 302}};
Line(311) = {{205, 305}};
Curve Loop(303) = {{203, 310, -303, -308}};
Surface(303) = {{303}};
Curve Loop(304) = {{204, 309, -304, -310}};
Surface(304) = {{304}};
Curve Loop(305) = {{201, 311, -301, -309}};
Surface(305) = {{305}};
Curve Loop(306) = {{202, 308, -302, -311}};
Surface(306) = {{306}};
Curve Loop(307) = {{205, 309, -305, -307}};
Plane Surface(307) = {{307}};
Curve Loop(308) = {{206, 308, -306, -307}};
Plane Surface(308) = {{308}};
Surface Loop(301) = {{202, 303, 304, 307, 308, 302}};
Volume(301) = {{301}};
Surface Loop(302) = {{307, 308, 301, 306, 305, 301}};
Volume(302) = {{302}};

{ELECTRODES}

Physical Volume("Core", 60) = {{201, 202}};
Physical Volume("Padding", 59) = {{101, 102, 301, 302}};
Physical Surface("Top", 61) = {{301,302}};
Physical Surface("Bottom", 62) = {{1,2}};

