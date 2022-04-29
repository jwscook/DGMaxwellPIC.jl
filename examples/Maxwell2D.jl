using DGMaxwellPIC, Plots, TimerOutputs, StaticArrays, LinearAlgebra

const NX = 32;
const NY = 4;

const OX = 2;
const OY = 2;

const state2D = State([OX, OY], LobattoNodes);

const DIMS = 2

const a = zeros(DIMS);# randn(DIMS);
const b = ones(DIMS);#a .+ rand(DIMS) .* 10;
const area = prod(b .- a)

gridposition(x) = SVector{DIMS, Float64}((x .* (b .- a) .+ a))

const grid2D = Grid([Cell(deepcopy(state2D), gridposition(((i-1)/NX, (j-1)/NY)), gridposition((i/NX, j/NY))) for i in 1:NX, j in 1:NY]);

#magneticfield!(grid2D, x->exp(-(sum(x-(b+a)/2)).^2 * 10), 3);

const s0 = DGMaxwellPIC.speedoflight
const k = 4pi / (b[1] - a[1])
const ω = s0 * k

fB(x, t=0) = sin(k * x[1] - ω * t)
fE(x, t=0) = s0 * fB(x, t)

electricfield!(grid2D, fE, 1);
electricfield!(grid2D, fE, 2);
magneticfield!(grid2D, fB, 3);

const dtc = minimum((b .- a)./(NX, NY)./(OX, OY)) / DGMaxwellPIC.speedoflight
const dt = dtc * 0.5

@show "Assembling"
const M = assemble(grid2D, upwind=0.0) * dt / 2;
@show "Building Crank-Nicolson"
const A = (I - M) \ Matrix(I + M);
@show "Calcuating sources"
const S = sources(grid2D);
@show "Fetching dofs vector"
const u = dofs(grid2D);


const to = TimerOutput()
@gif for i in 1:100
  @timeit to "u .+=" u .+= A * u
  @timeit to "dofs!" dofs!(grid2D, u)
  p1 = heatmap(electricfield(grid2D, 1))
  p2 = heatmap(electricfield(grid2D, 2))
  p3 = heatmap(electricfield(grid2D, 3))
  p4 = heatmap(magneticfield(grid2D, 1))
  p5 = heatmap(magneticfield(grid2D, 2))
  p6 = heatmap(magneticfield(grid2D, 3))
  plot(p1, p2, p3, p4, p5, p6, layout = (@layout [a b c; d e f]))
  @show i
end every 2
show(to)
