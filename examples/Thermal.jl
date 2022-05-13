using DGMaxwellPIC, Plots, TimerOutputs, StaticArrays

const NX = 32;
const NY = 16;

const OX = 3;
const OY = 5;

const state2D = State([OX, OY], LegendreNodes);

const DIMS = 2

const a = zeros(DIMS);# randn(DIMS);
const b = ones(DIMS);#a .+ rand(DIMS) .* 10;
const area = prod(b .- a)
gridposition(x) = SVector{DIMS, Float64}((x .* (b .- a) .+ a))
const grid2D = Grid([Cell(deepcopy(state2D), gridposition(((i-1)/NX, (j-1)/NY)), gridposition((i/NX, j/NY))) for i in 1:NX, j in 1:NY]);

magneticfield!(grid2D, x->1.0, 3); # set z field to 1

@show magneticfield(grid2D, rand(2) .* (b .- a) .+ a)

@show DGMaxwellPIC.divB(grid2D, rand(2) .* (b .- a) .+ a)

@show magneticfield(Cell(deepcopy(state2D), a, b), rand(2) .* (b .- a) .+ a)

function distributionfunction(xv)
  x = xv[1:2]
  v = xv[3:5]
  return exp(-sum(v.^2))
end
const NP = NX * NY * OX * OY * 10
@show NP
const particledata = DGMaxwellPIC.ParticleData(distributionfunction, NP, [a..., -4, -4, -4], [b..., 4, 4, 4]);
weight!(particledata, 32pi * area / length(particledata));

@show magneticfield(grid2D, rand(2) .* (b .- a) .+ a)
@show electricfield(grid2D, rand(2) .* (b .- a) .+ a)

const plasma = Plasma([Species(particledata, charge=1.0, mass=1.0)]);
sort!(plasma, grid2D) # sort particles by cellid

const A = assemble(grid2D);

const u = dofs(grid2D);
const S = deepcopy(u);

Bz = magneticfield(grid2D, 3);

const dtc = minimum((b .- a)./(NX, NY)) / DGMaxwellPIC.speedoflight
const dt = dtc *0.5

const grid2Dcopy = deepcopy(grid2D)
sort!(plasma, grid2Dcopy)

using StatProfilerHTML, Profile
begin 
@time depositcurrent!(grid2Dcopy, plasma)
Profile.clear()
@profilehtml depositcurrent!(grid2Dcopy, plasma)
end
dofs!(grid2D, 0.0)


const to = TimerOutput()
@gif for i in 1:100
  @timeit to "deposit" depositcurrent!(grid2D, plasma)
  @timeit to "sources!" sources!(S, grid2D)
  @timeit to "u .+=" u .+= (A * u .+ S) .* dt
  @timeit to "dofs!" dofs!(grid2D, u)
  advance!(plasma, grid2D, dt)
  p1 = heatmap(electricfield(grid2D, 1))
  p2 = heatmap(electricfield(grid2D, 2))
  p3 = heatmap(electricfield(grid2D, 3))
  p4 = heatmap(magneticfield(grid2D, 1))
  p5 = heatmap(magneticfield(grid2D, 2))
  p6 = heatmap(magneticfield(grid2D, 3))
  plot(p1, p2, p3, p4, p5, p6, layout = (@layout [a b c; d e f]))
  @show i
end
show(to)
