Mesh.MshFileVersion = 2.2;

TreeDiameter = {TreeDiameter};
CoreThickness = {CoreThickness};
Padding = {Padding};

MeshSizeCore = {MeshSizeCore};
MeshSizePadding = {MeshSizePadding};
MeshSizeElectrodes = {MeshSizeElectrodes};

//bottom padding:
Point(1) = {{0,               0   , -CoreThickness/2 - Padding, MeshSizePadding}};
Point(3) = {{TreeDiameter/2 , 0   , -CoreThickness/2 - Padding, MeshSizePadding}};
Point(2) = {{0              , 2*Pi, -CoreThickness/2 - Padding,  MeshSizePadding}};
Point(4) = {{TreeDiameter/2 ,  2*Pi, -CoreThickness/2 - Padding, MeshSizePadding}};
Line(1) = {{1, 3}};
Line(2) = {{3, 4}};
Line(3) = {{4, 2}};
Line(4) = {{2, 1}};

Curve Loop(1) = {{1, 2, 3, 4}};
Plane Surface(1) = {{-1}};

// bollom core
Point(201) = {{0,               0   , -CoreThickness/2, MeshSizeCore}};
Point(203) = {{TreeDiameter/2 , 0   , -CoreThickness/2, MeshSizeCore}};
Point(202) = {{0              , 2*Pi, -CoreThickness/2, MeshSizeCore}};
Point(204) = {{TreeDiameter/2 ,  2*Pi, -CoreThickness/2, MeshSizeCore}};
Line(201) = {{201, 203}};
Line(202) = {{203, 204}};
Line(203) = {{204, 202}};
Line(204) = {{202, 201}};

Curve Loop(201) = {{201, 202, 203, 204}};
Plane Surface(201) = {{201}};

// top core
Point(301) = {{0,               0   ,  CoreThickness/2, MeshSizeCore}};
Point(303) = {{TreeDiameter/2 , 0   ,  CoreThickness/2, MeshSizeCore}};
Point(302) = {{0              , 2*Pi,  CoreThickness/2, MeshSizeCore}};
Point(304) = {{TreeDiameter/2 ,  2*Pi, CoreThickness/2, MeshSizeCore}};
Line(301) = {{301, 303}};
Line(302) = {{303, 304}};
Line(303) = {{304, 302}};
Line(304) = {{302, 301}};

Curve Loop(301) = {{301, 302, 303, 304}};
Plane Surface(301) = {{301}};

//top padding:
Point(401) = {{0,               0   , CoreThickness/2 + Padding, MeshSizePadding}};
Point(403) = {{TreeDiameter/2 , 0   , CoreThickness/2 + Padding, MeshSizePadding}};
Point(402) = {{0              , 2*Pi, CoreThickness/2 + Padding, MeshSizePadding}};
Point(404) = {{TreeDiameter/2 ,  2*Pi, CoreThickness/2 + Padding, MeshSizePadding}};
Line(401) = {{401, 403}};
Line(402) = {{403, 404}};
Line(403) = {{404, 402}};
Line(404) = {{402, 401}};

Curve Loop(401) = {{401, 402, 403, 404}};
Plane Surface(401) = {{401}};

Line(405) = {{1, 201}};
Line(406) = {{3, 203}};
Line(407) = {{4, 204}};
Line(408) = {{2, 202}};
Line(409) = {{201, 301}};
Line(410) = {{203, 303}};
Line(411) = {{204, 304}};
Line(412) = {{202, 302}};
Line(413) = {{301, 401}};
Line(414) = {{303, 403}};
Line(415) = {{304, 404}};
Line(416) = {{302, 402}};

Curve Loop(402) = {{1, 406, -201, -405}};
Plane Surface(402) = {{402}};
Curve Loop(403) = {{2, 407, -202, -406}};
Plane Surface(403) = {{403}};
Curve Loop(404) = {{3, 408, -203, -407}};
Plane Surface(404) = {{404}};
Curve Loop(405) = {{4, 405, -204, -408}};
Plane Surface(405) = {{405}};
Curve Loop(406) = {{201, 410, -301, -409}};
Plane Surface(406) = {{406}};
Curve Loop(407) = {{202, 411, -302, -410}};
Plane Surface(407) = {{407}};
Curve Loop(408) = {{203, 412, -303, -411}};
Plane Surface(408) = {{408}};
Curve Loop(409) = {{204, 409, -304, -412}};
Plane Surface(409) = {{409}};
Curve Loop(410) = {{301, 414, -401, -413}};
Plane Surface(410) = {{410}};
Curve Loop(411) = {{302, 415, -402, -414}};
Plane Surface(411) = {{411}};
Curve Loop(412) = {{303, 416, -403, -415}};
Plane Surface(412) = {{412}};
Curve Loop(413) = {{304, 413, -404, -416}};
Plane Surface(413) = {{413}};

Surface Loop(1) = {{1, 402, 403, 404, 405, 201}};
Volume(1) = {{1}};
Surface Loop(2) = {{201, 406, 407, 408, 409, 301}};
Volume(2) = {{2}};
Surface Loop(3) = {{401, 410, 411, 412, 413, 301}};
Volume(3) = {{3}};

{ELECTRODES}

Physical Volume("Core", 60) = {{2}};
Physical Volume("Padding", 59) = {{1, 3}};
Physical Surface("Top", 61) = {{401}};
Physical Surface("Bottom", 62) = {{1}};

