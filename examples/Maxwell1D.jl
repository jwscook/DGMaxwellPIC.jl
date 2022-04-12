using DGMaxwellPIC, Plots, TimerOutputs, StaticArrays

const NX = 16;

const OX = 5;

const state2D = State([OX], LegendreNodes);

const DIMS = 1
const L = 1

const a = zeros(DIMS);
const b = ones(DIMS) .* L;
const area = prod(b .- a)

gridposition(x) = SVector{DIMS, Float64}((x .* (b .- a) .+ a))

const grid2D = Grid([Cell(deepcopy(state2D), gridposition((i-1)/NX), gridposition(i/NX)) for i in 1:NX]);

const s0 = DGMaxwellPIC.speedoflight
const k = 4 * pi
const ω = s0 * k

fBz(x, t=0) = sin(x[1] * k - ω  * t)
fEy(x, t=0) = s0 * fBz(x, t)

electricfield!(grid2D, fEy, 2);
magneticfield!(grid2D, fBz, 3);

const A = assemble(grid2D, upwind=0.0);
const S = sources(grid2D);
const u = dofs(grid2D);

const dtc = minimum((b .- a)./NX./OX) / s0
const dt = dtc * 0.05

const to = TimerOutput()
const x = collect(1/NX/2:1/NX:1-1/NX/2) .* L

const nturns = 0.1
const NI = Int(ceil(nturns * L / s0 / dt))

@gif for i in 1:NI
  @timeit to "u .+=" u .+= A * u .* dt
  @timeit to "dofs!" dofs!(grid2D, u)
  t = i * dt
  p1 = plot(x, electricfield(grid2D, 1), ylims=[-s0,s0]); title!("$i of $NI")
  p2 = plot(x, electricfield(grid2D, 2), ylims=[-s0,s0])
  plot!(p2, x, [fEy([xi], t) for xi in x], ylims=[-s0,s0])
  p3 = plot(x, electricfield(grid2D, 3), ylims=[-s0,s0])
  p4 = plot(x, magneticfield(grid2D, 1), ylims=[-1,1])
  p5 = plot(x, magneticfield(grid2D, 2), ylims=[-1,1])
  p6 = plot(x, magneticfield(grid2D, 3), ylims=[-1,1])
  plot!(p6, x, [fBz([xi], t) for xi in x], ylims=[-1,1])
  plot(p1, p2, p3, p4, p5, p6, layout = (@layout [a b c; d e f]))
  @show i, i * dt * s0
end every 1
show(to)
