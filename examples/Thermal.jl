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


@show DGMaxwellPIC.divB(grid2D, rand(2) .* (b .- a) .+ a)
@show DGMaxwellPIC.divB(grid2D, rand(2) .* (b .- a) .+ a)
@show DGMaxwellPIC.divB(grid2D, rand(2) .* (b .- a) .+ a)

@show magneticfield(cell, rand(2) .* (b .- a) .+ a)

function distributionfunction(xv)
  x = xv[1:2]
  v = xv[3:5]
  return exp(-sum(v.^2))
end

particledata = DGMaxwellPIC.ParticleData(distributionfunction, 10_000, [a..., -6, -6, -6], [b..., 6, 6, 6])
weight!(particledata, 1)
@show magneticfield(grid2D, rand(2) .* (b .- a) .+ a)
@show electricfield(grid2D, rand(2) .* (b .- a) .+ a)

const plasma = Plasma([Species(particledata, charge=1.0, mass=1.0)])

const A = assemble(grid2D)
const S = sources(grid2D)

u = dofs(grid2D)

dt = 0.01

for i in 1:100
  depositcurrent!(grid2D, plasma)
  S .= sources(grid2D)
  u .+= (A * u .+ S) .* dt
  dofs!(grid2D, u)
  push!(plasma, grid2D, dt)
end

