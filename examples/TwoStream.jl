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

function distributionfunction(xv)
  x = xv[1:2]
  v = xv[3:5]
  return exp(-sum(v.^2))
end
const NP = NX * NY * OX * OY * 10

const dataxvw = zeros(DIMS + 3 + 1, NP);
dataxvw[1:DIMS, :] .= rand(DIMS, NP) .* (b .- a) .+ a
dataxvw[DIMS+1, :] .= rand((-1, 1), NP)
const particledata = DGMaxwellPIC.ParticleData(dataxvw);
weight!(particledata, 32pi * area / length(particledata));

const plasma = Plasma([Species(particledata, charge=1.0, mass=1.0)]);
sort!(plasma, grid2D) # sort particles by cellid

const A = assemble(grid2D);
const S = sources(grid2D);
const u = dofs(grid2D);

const dtc = minimum((b .- a)./(NX, NY)) / DGMaxwellPIC.speedoflight
const dt = dtc *0.5

const grid2Dcopy = deepcopy(grid2D)
sort!(plasma, grid2Dcopy)

const to = TimerOutput()
@gif for i in 1:100
  @timeit to "advance!" advance!(plasma, grid2D, dt/2)
  @timeit to "deposit" depositcurrent!(grid2D, plasma)
  @timeit to "source" S .= sources(grid2D)
  @timeit to "u .+=" u .+= (A * u .+ S) .* dt
  @timeit to "dofs!" dofs!(grid2D, u)
  @timeit to "advance!" advance!(plasma, grid2D, dt/2)
  p1 = heatmap(electricfield(grid2D, 1))
  p2 = heatmap(electricfield(grid2D, 2))
  p3 = heatmap(electricfield(grid2D, 3))
  p4 = heatmap(magneticfield(grid2D, 1))
  p5 = heatmap(magneticfield(grid2D, 2))
  p6 = heatmap(magneticfield(grid2D, 3))
  plot(p1, p2, p3, p4, p5, p6, layout = (@layout [a b c; d e f]))
  @show i
end every 10
show(to)
