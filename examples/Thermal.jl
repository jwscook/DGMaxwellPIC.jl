using DGMaxwellPIC

const NX = 32
const NY = 16

const OX = 3
const OY = 5

const state2D = State([OX, OY], LagrangeOrthogonal)

const grid2D = Grid([Cell(deepcopy(state2D), ((i-1)/NX, (j-1)/NY), (i/NX, j/NY)) for i in 1:NX, j in 1:NY])

magneticfield!(grid2D, x->1.0, 3) # set z field to 1

@show magneticfield(grid2D, rand(2))

a = randn(2)
b = a .+ rand(2)
cell = Cell(deepcopy(state2D), a, b)
magneticfield!(cell, x->1, 3)
@show magneticfield(cell, rand(2) .* (b .- a) .+ a)

function distributionfunction(xv)
  x = xv[1:2]
  v = xv[3:5]
  return exp(-sum(v.^2))
end

particles = DGMaxwellPIC.particlephasepositions(distributionfunction, 10_000, [a..., -6, -6, -6], [b..., 6, 6, 6])


