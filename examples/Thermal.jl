using DGMaxwellPIC, Plots, TimerOutputs

const NX = 32;
const NY = 16;

const OX = 3;
const OY = 5;

const state2D = State([OX, OY], LagrangeOrthogonal);

const a = randn(2);
const b = a .+ rand(2) .* 10;
const area = prod(b .- a)
gridposition(x) = x .* (b .- a) .+ a
const grid2D = Grid([Cell(deepcopy(state2D), gridposition(((i-1)/NX, (j-1)/NY)), gridposition((i/NX, j/NY))) for i in 1:NX, j in 1:NY]);

magneticfield!(grid2D, x->1.0, 3); # set z field to 1

@show magneticfield(grid2D, rand(2))

@show DGMaxwellPIC.divB(grid2D, rand(2) .* (b .- a) .+ a)
@show DGMaxwellPIC.divB(grid2D, rand(2) .* (b .- a) .+ a)
@show DGMaxwellPIC.divB(grid2D, rand(2) .* (b .- a) .+ a)

@show magneticfield(Cell(deepcopy(state2D), a, b), rand(2) .* (b .- a) .+ a)

function distributionfunction(xv)
  x = xv[1:2]
  v = xv[3:5]
  return exp(-sum(v.^2))
end

const particledata = DGMaxwellPIC.ParticleData(distributionfunction, 100_000, [a..., -4, -4, -4], [b..., 4, 4, 4]);
weight!(particledata, 32pi * area / length(particledata));

@show magneticfield(grid2D, rand(2) .* (b .- a) .+ a)
@show electricfield(grid2D, rand(2) .* (b .- a) .+ a)

const plasma = Plasma([Species(particledata, charge=1.0, mass=1.0)]);

const A = assemble(grid2D);
const S = sources(grid2D);

const u = dofs(grid2D);

Bz = magneticfield(grid2D, 3);

const dt = 0.001

  @time depositcurrent!(grid2D, plasma)
  @time depositcurrent!(grid2D, plasma)

#const to = TimerOutput()
#@gif for i in 1:1
#  @timeit to "deposit" depositcurrent!(grid2D, plasma)
#  @timeit to "source" S .= sources(grid2D)
#  @timeit to "u .+=" u .+= (A * u .+ S) .* dt
#  @timeit to "dofs!" dofs!(grid2D, u)
#  advance!(plasma, grid2D, dt)
#  p1 = heatmap(electricfield(grid2D, 1))
#  p2 = heatmap(electricfield(grid2D, 2))
#  p3 = heatmap(electricfield(grid2D, 3))
#  p4 = heatmap(magneticfield(grid2D, 1))
#  p5 = heatmap(magneticfield(grid2D, 2))
#  p6 = heatmap(magneticfield(grid2D, 3))
#  l = @layout [a b c; d e f]
#  plot(p1, p2, p3, p4, p5, p6, layout = l)
#  @show i
#end
#show(to)
