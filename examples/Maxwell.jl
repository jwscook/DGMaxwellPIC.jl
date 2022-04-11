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

magneticfield!(grid2D, x->exp(-(sum(x-(b+a)/2)).^2 * 10), 3);

const A = assemble(grid2D);
const S = sources(grid2D);
const u = dofs(grid2D);

const dtc = minimum((b .- a)./(NX, NY)./(OX, OY)) / DGMaxwellPIC.speedoflight
const dt = dtc * 0.5


const to = TimerOutput()
@gif for i in 1:1000
  @timeit to "u .+=" u .+= A * u .* dt
  @timeit to "dofs!" dofs!(grid2D, u)
  p1 = heatmap(electricfield(grid2D, 1))
  p2 = heatmap(electricfield(grid2D, 2))
  p3 = heatmap(electricfield(grid2D, 3))
  p4 = heatmap(magneticfield(grid2D, 1))
  p5 = heatmap(magneticfield(grid2D, 2))
  p6 = heatmap(magneticfield(grid2D, 3))
  plot(p1, p2, p3, p4, p5, p6, layout = (@layout [a b c; d e f]))
  @show i
end every 100
show(to)
