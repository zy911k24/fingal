# ERT on Trees

Sensitivity of a ring of electrodes on a tree trunk, for wood with cylindrical
anisotropy. The trunk is a cylinder of diameter `TreeDiameter` with padding above and
below; 48 electrodes sit on one ring at mid-height, and the schedule is a Wenner array
that wraps around the ring (dipole spacing `a = 1 … 15`, giving 720 quadrupoles).

The conductivity is diagonal in the cylindrical frame of the trunk,

    sigma = sigma_r * e_r x e_r + sigma_t * e_t x e_t + sigma_z * e_z x e_z

with, by default, `sigma_r = 5e-4`, `sigma_t = 2e-4`, `sigma_z = 1e-3` S/m.

The example is solved twice, in two independent formulations, so each can be checked
against the other:

| | mesh | anisotropy | script |
|---|---|---|---|
| cartesian | unstructured tets, `tree.fly` | full rotating tensor | [`sensWenner.py`](./sensWenner.py) |
| cylindrical | structured `Brick` in `(r,t,z)`, built in the script | diagonal, grid aligned | [`sensCWenner.py`](./sensCWenner.py) |


## Cartesian run

Generate the mesh from [`treegeoxyz.tpl`](./treegeoxyz.tpl):

    ./mkMesh.py

This writes `tree.geo`, meshes it with `gmsh`, and converts to `tree.fly`. It also
writes `stations.csv` with the electrode positions in **cartesian** `(x,y,z)`. With the
default settings the mesh has about 440k nodes and takes gmsh about a minute.

Then run the sensitivity:

    ./sensWenner.py

Output goes to `resultAnisoSmallXYZ_R0.silo` (about 12 minutes).

`sensWenner.py` asserts that the stations form a ring in the `x-y` plane before it
starts. This catches a stale cylindrical `stations.csv`/`tree.fly` pair, which would
otherwise be read as cartesian without complaint and silently produce nonsense.


## Cylindrical run

    ./sensCWenner.py

No mesh file is involved: the script builds its own `Brick` on the computational box
`(r, t, z)` with `periodic1=True`, and maps the coordinates to cartesian just before
writing `resultAnisoSmallCyl.silo`. This is the slower of the two runs.

Three things about this formulation are easy to get wrong, and all three change the
result qualitatively rather than by a small factor.

**The metric.** With `grad_x(u) = (du/dr, (1/r) du/dt, du/dz)` and `dx dy dz = r dr dt dz`,

    int grad(v).sigma.grad(u) dxdydz
       = int ( sigma_r v_r u_r + sigma_t r^-2 v_t u_t + sigma_z v_z u_z ) * r dr dt dz

so the PDE coefficient on the computational box is

    A = diag( sigma_r * r ,  sigma_t / r ,  sigma_z * r )

The `1/r` is what couples the nodes on the axis. Without it those nodes stay
independent degrees of freedom and `u` becomes multivalued at `r=0` — the axis is
effectively a free surface and the model has no pith at all.

**The source.** The electrode ring sits on the bottom face `z=0`, which carries no
Dirichlet condition and is therefore the mirror plane of the physical problem: the
domain solved is the half `z>0`. Only half the injected current flows into it, so a
unit physical current is a source of `1/2` (`SOURCE_STRENGTH`). Using `1` instead makes
every potential, and hence every sensitivity, a factor 2 too large. The apparent
conductivity `sigma_a = f_geo/F` is a ratio and is unaffected either way.

**The output.** The sensitivity kernels are derivatives of the bilinear form above and
are therefore densities with respect to `dr dt dz`:

    d/d_sigma_r -> r * u_r v_r        d/d_sigma_t -> (1/r) * u_t v_t        d/d_sigma_z -> r * u_z v_z

The script divides them by `r` before saving so that what is written is a density per
*physical* volume — the same quantity `sensWenner.py` computes, and what an inversion
cell weights. Plotting the unscaled kernels instead multiplies the axis behaviour by
`r` and hides it completely.

Note also that as *densities* the two scripts agree pointwise, but `sensCWenner.py`
integrates over `z>0` only. Summing its kernel over cells to get `dF/d_sigma` for a
mirror-symmetric perturbation needs the factor 2 back, since the material at `z>0` and
its image move together.


## The axis, and `R0`

The fibre direction `e_r` is undefined at `r=0`, so the tensor above is genuinely
discontinuous on the trunk axis. Separating `u ~ r^alpha cos(m t)` near the axis gives

    sigma_r * alpha^2 = sigma_t * m^2      ->      alpha = m * sqrt(sigma_t/sigma_r)

For the default values `alpha = 0.632` at `m=1`, so `|grad u| ~ r^(alpha-1)` diverges
and the sensitivity density goes as `r^(2*alpha-2) = r^-0.74`. This is a property of the
model, not of the discretisation: refining the mesh makes the peak grow, not shrink.
The energy is finite (`u` is in `H1`), so the solution away from the axis is fine.

The cure is to stop claiming a radial direction exists at the centre. `sensWenner.py`
softens the normalisation with a pith radius `R0`:

    e = x[:2]/sqrt(x[0]**2 + x[1]**2 + R0**2)
    A_plane = sigma_t * I + (sigma_r - sigma_t) * e x e

so `|e|^2 = r^2/(r^2+R0^2)`. For `r >> R0` this is the old unit vector; at `r=0` the
tensor is `sigma_t * I`, isotropic and non-degenerate. The eigenvalues are
`sigma_t + (sigma_r-sigma_t)*r^2/(r^2+R0^2)` and `sigma_t`, so the tensor is uniformly
positive definite. The kernels follow from `dA/d_sigma_r = e x e` and
`dA/d_sigma_t = I - e x e`; note the tangential kernel has to be written as the in-plane
product minus the radial one, since the shorter form `(e0*g1 - e1*g0)` is the tangential
component only while `|e| = 1`.

`R0` needs a few elements across it, so [`treegeoxyz.tpl`](./treegeoxyz.tpl) refines
along the axis with a gmsh distance field on `Line(207)`, controlled by
`AxisRefineRadius` and `MeshSizeAxis` in [`mkMesh.py`](./mkMesh.py). `SizeMax` is set to
`MeshSizePadding`, the coarsest size already in the model: gmsh combines the background
field with the point sizes by taking the minimum, so the field can only refine near the
axis and never touches the padding. With the defaults this takes the element size at
the axis from `1.9e-2` to `4.2e-3` at a cost of 1.7x in node count, which makes
`R0 = 0.01` resolvable. A millimetre-scale pith would need a smaller `MeshSizeAxis` and
a correspondingly smaller `AxisRefineRadius`.

`sensCWenner.py` has no `R0`: it still carries the singular fibre field, which is why
the two runs are expected to differ inside `r ~ R0`.


## Cross validation

Both runs, restricted to the same slab `z` in `[0, 0.125]`, volume weighted, per
physical volume and per unit injected current:

```
                    s_r                        s_t                        s_z
  r-bin          CYL     XYZ_R0  ratio     CYL     XYZ_R0  ratio    CYL    XYZ_R0  ratio
 [0.000,0.016)  3.865e4  2.240e4  1.73   1.320e5  1.421e5  0.93    15.5    68.7   0.23
 [0.016,0.031)  1.931e4  1.612e4  1.20   5.374e4  5.876e4  0.92    42.8    59.5   0.72
 [0.031,0.047)  1.237e4  1.098e4  1.13   3.898e4  4.040e4  0.96    77.6    82.5   0.94
 [0.047,0.062)  8.963e3  8.172e3  1.10   3.323e4  3.371e4  0.99   116.8   116.2   1.01
 [0.062,0.078)  6.930e3  6.422e3  1.08   3.085e4  3.094e4  1.00   161.3   156.6   1.03
 [0.078,0.094)  5.600e3  5.253e3  1.07   3.017e4  3.007e4  1.00   213.1   204.7   1.04
 [0.094,0.109)  4.691e3  4.449e3  1.06   3.055e4  3.035e4  1.01   275.6   263.4   1.05
 [0.109,0.125)  4.061e3  3.898e3  1.04   3.178e4  3.153e4  1.01   354.4   337.3   1.05
```

Outside the pith the two agree in absolute terms — `s_t` to 1%, `s_z` to 1-5%, `s_r` to
4-10% — on different meshes, different domains and different boundary treatments. Part
of the `s_r` bias is `R0` itself: the kernel carries `|e|^2 = r^2/(r^2+R0^2)`, worth
3.2% at `r = 0.055` and 0.7% at `r = 0.117`. The apparent conductivities agree to 0.5%
for `a >= 3`.

Inside `r < 0.03` the two part company by design, since only the cartesian run has `R0`.

The profiles above were extracted with VisIt in batch, binning on

    rad = recenter(sqrt(coord(Elements)[0]^2 + coord(Elements)[1]^2), "zonal")

and querying `Average Value` under a `Threshold` on `rad` and on the `z` coordinate.


## Files

    mkMesh.py          cartesian mesh generator (uses treegeoxyz.tpl)
    treegeoxyz.tpl     cartesian (x,y,z) gmsh template, with the axis refinement field
    sensWenner.py      cartesian sensitivity run, has R0
    sensCWenner.py     cylindrical sensitivity run, builds its own periodic Brick

    mkMeshCyl.py       OBSOLETE
    treegeocyl.tpl     OBSOLETE

`mkMeshCyl.py` built the cylindrical box with gmsh, but the `t` direction has to be
periodic and `ReadGmsh` does not carry periodicity over, so the `t=0` and `t=2*Pi` faces
stay unconnected. The cylindrical formulation therefore builds its own `Brick` instead.
Note that `mkMeshCyl.py` writes `tree.geo`/`tree.msh`/`tree.fly`/`stations.csv`, the
same names `mkMesh.py` uses — running it leaves `sensWenner.py` reading a cylindrical
mesh as if it were cartesian, which is what the ring-radius assertion now catches.
