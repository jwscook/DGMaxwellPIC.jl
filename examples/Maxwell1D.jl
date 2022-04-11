using DGMaxwellPIC, Plots, TimerOutputs, StaticArrays

const NX = 32;

const OX = 4;

const state2D = State([OX], LegendreNodes);

const DIMS = 1

const a = zeros(DIMS);
const b = ones(DIMS);
const area = prod(b .- a)

gridposition(x) = SVector{DIMS, Float64}((x .* (b .- a) .+ a))

const grid2D = Grid([Cell(deepcopy(state2D), gridposition((i-1)/NX), gridposition(i/NX)) for i in 1:NX]);

const s0 = DGMaxwellPIC.speedoflight
electricfield!(grid2D, x->s0 * sin(x[1] * 4 * 2pi), 2);
magneticfield!(grid2D, x->sin(x[1] * 4 * 2pi), 3);

const A = assemble(grid2D, upwind=1.0);
const S = sources(grid2D);
const u = dofs(grid2D);

const dtc = minimum((b .- a)./NX./OX) / DGMaxwellPIC.speedoflight
const dt = dtc * 0.5

const to = TimerOutput()
const NI = 1000
@gif for i in 1:NI
  @timeit to "u .+=" u .+= A * u .* dt
  @timeit to "dofs!" dofs!(grid2D, u)
  p1 = plot(electricfield(grid2D, 1), ylims=[-s0,s0]); title!("$i of $NI")
  p2 = plot(electricfield(grid2D, 2), ylims=[-s0,s0])
  p3 = plot(electricfield(grid2D, 3), ylims=[-s0,s0])
  p4 = plot(magneticfield(grid2D, 1), ylims=[-1,1])
  p5 = plot(magneticfield(grid2D, 2), ylims=[-1,1])
  p6 = plot(magneticfield(grid2D, 3), ylims=[-1,1])
  plot(p1, p2, p3, p4, p5, p6, layout = (@layout [a b c; d e f]))
  @show i, i * dt / s0
end every 1
show(to)
