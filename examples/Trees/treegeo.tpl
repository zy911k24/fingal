TreeDiameter = {TreeDiameter};
CoreThickness = {CoreThickness};
Padding = {Padding};

MeshSizeCore = {MeshSizeCore};
MeshSizePadding = {MeshSizePadding};
MeshSizeElectrodes = {MeshSizeElectrodes};


Point(1) = {{0, -0, -CoreThickness/2, MeshSizeCore}};
Point(3) = {{-TreeDiameter/2, 0, -CoreThickness/2, MeshSizeCore}};
Point(2) = {{0, -TreeDiameter/2,  -CoreThickness/2, MeshSizeCore}};
Point(4) = {{TreeDiameter/2, 0, -CoreThickness/2, MeshSizeCore}};
Point(5) = {{0, TreeDiameter/2, -CoreThickness/2, MeshSizeCore}};
Circle(1) = {{4, 1, 5}};
Circle(2) = {{5, 1, 3}};
Circle(3) = {{3, 1, 2}};
Circle(4) = {{2, 1, 4}};
Curve Loop(1) = {{4, 1, 2, 3}};
Plane Surface(1) = {{-1}};


Point(11) = {{0, -0,  CoreThickness/2, MeshSizeCore}};
Point(13) = {{-TreeDiameter/2, 0, CoreThickness/2, MeshSizeCore}};
Point(12) = {{0, -TreeDiameter/2,  CoreThickness/2, MeshSizeCore}};
Point(14) = {{TreeDiameter/2, 0, CoreThickness/2, MeshSizeCore}};
Point(15) = {{0, TreeDiameter/2, CoreThickness/2, MeshSizeCore}};
Circle(11) = {{14, 11, 15}};
Circle(12) = {{15, 11, 13}};
Circle(13) = {{13, 11, 12}};
Circle(14) = {{12, 11, 14}};
Curve Loop(11) = {{14, 11, 12, 13}};
Plane Surface(11) = {{11}};

// CORE REGION
Line(15) = {{2, 12}};
Line(16) = {{3, 13}};
Line(17) = {{5, 15}};
Line(18) = {{4, 14}};

Curve Loop(12) = {{18, -14, -15, 4}};
Surface(12) = {{12}};
Curve Loop(13) = {{15, -13, -16, 3}};
Surface(13) = {{13}};
Curve Loop(14) = {{16, -12, -17, 2}};
Surface(14) = {{14}};
Curve Loop(15) = {{17, -11, -18, 1}};
Surface(15) = {{15}};

Surface Loop(1) = {{11, 13, 1, 12, 15, 14}};
Volume(1) = {{1}};

// padding south
Point(31) = {{0, -0, -CoreThickness/2 - Padding, MeshSizePadding}};
Point(33) = {{-TreeDiameter/2, 0, -CoreThickness/2 - Padding, MeshSizePadding}};
Point(32) = {{0, -TreeDiameter/2,  -CoreThickness/2 - Padding, MeshSizePadding}};
Point(34) = {{TreeDiameter/2, 0, -CoreThickness/2 - Padding, MeshSizePadding}};
Point(35) = {{0, TreeDiameter/2, -CoreThickness/2 - Padding , MeshSizePadding}};
Circle(31) = {{34, 31, 35}};
Circle(32) = {{35, 31, 33}};
Circle(33) = {{33, 31, 32}};
Circle(34) = {{32, 31, 34}};
Curve Loop(31) = {{34, 31, 32, 33}};
Plane Surface(31) = {{-31}};

Line(45) = {{34, 4}};
Line(46) = {{32, 2}};
Line(47) = {{35, 5}};
Line(48) = {{33, 3}};
Curve Loop(42) = {{46, 4, -45, -34}};
Surface(42) = {{-42}};

Curve Loop(43) = {{3, -46, -33, 48}};
Surface(43) = {{-43}};

Curve Loop(44) = {{32, 48, -2, -47}};
Surface(44) = {{44}};

Curve Loop(45) = {{47, -1, -45, 31}};
Surface(45) = {{45}};

Surface Loop(3) = {{43, 42, 45, 31, 44, 1}};
Volume(3) = {{3}};

// padding north

Point(41) = {{0, -0,  CoreThickness/2 + Padding, MeshSizePadding}};
Point(43) = {{-TreeDiameter/2, 0, CoreThickness/2 + Padding, MeshSizePadding}};
Point(42) = {{0, -TreeDiameter/2,  CoreThickness/2 + Padding, MeshSizePadding}};
Point(44) = {{TreeDiameter/2, 0, CoreThickness/2 + Padding, MeshSizePadding}};
Point(45) = {{0, TreeDiameter/2, CoreThickness/2 + Padding, MeshSizePadding}};
Circle(41) = {{44, 41, 45}};
Circle(42) = {{45, 41, 43}};
Circle(43) = {{43, 41, 42}};
Circle(44) = {{42, 41, 44}};
Curve Loop(41) = {{44, 41, 42, 43}};
Plane Surface(41) = {{41}};

Line(55) = {{44, 14}};
Line(56) = {{42, 12}};
Line(57) = {{45, 15}};
Line(58) = {{43, 13}};


Curve Loop(46) = {{13, -56, -43, 58}};
Surface(46) = {{46}};
Curve Loop(47) = {{58, -12, -57, 42}};
Surface(47) = {{-47}};
Curve Loop(48) = {{11, -57, -41, 55}};
Surface(48) = {{48}};
Curve Loop(49) = {{56, 14, -55, -44}};
Surface(49) = {{49}};

Surface Loop(2) = {{46, 47, 48, 49, 41, 11}};
Volume(2) = {{2}};

{ELECTRODES}

Physical Volume("Core", 60) = {{1}};
Physical Volume("Padding", 59) = {{1, 2, 3}};
//+
Physical Surface("Top", 61) = {{41}};
Physical Surface("Bottom", 62) = {{31}};
