using DGMaxwellPIC, Plots, TimerOutputs, StaticArrays, LinearAlgebra

const NX = 32;

const OX = 2;

const state2D = State([OX], LegendreNodes);

const DIMS = 1
const L = 10.0

const a = zeros(DIMS);
const b = ones(DIMS) .* L;
const area = prod(b .- a)

gridposition(x) = SVector{DIMS, Float64}((x .* (b .- a) .+ a))

const grid2D = Grid([Cell(deepcopy(state2D), gridposition((i-1)/NX), gridposition(i/NX)) for i in 1:NX]);

const s0 = DGMaxwellPIC.speedoflight
const k0 = 2 * pi / L
const k = 3 * k0
const ω = s0 * k

fBz(x, t=0) = sin(x[1] * k - ω * t)
fEy(x, t=0) = s0 * fBz(x, t)

electricfield!(grid2D, fEy, 2);
magneticfield!(grid2D, fBz, 3);
#DGMaxwellPIC.electricfielddofs!(grid2D, s0, 2);
#DGMaxwellPIC.magneticfielddofs!(grid2D, 1.0, 3);

const dtc = minimum((b .- a)./NX./OX) / s0
const dt = dtc * 0.2

# du/dt = A * u
# u1 - u0 = dt * (A * u)
# u1 - u0 = dt * (A * (u1 + u0)/2)
# u1 = u0 + dt/2 * A * u1 + dt/2 * A * u0
# (1 - dt/2 * A)*u1 = (1 + dt/2 * A) * u0
# u1 = (1 - dt/2 * A)^-1 (1 + dt/2 * A) * u0

const C = assemble(grid2D, upwind=0.0) * dt / 2;
const B = lu(I - C);
const A = B \ Matrix(I + C);
#const A = I + assemble(grid2D, upwind=0.0) * dt;
const S = sources(grid2D);
const u = dofs(grid2D);

const to = TimerOutput()
const x = collect(1/NX/2:1/NX:1-1/NX/2) .* L

const distance = 0.5 * L
const NI = Int(ceil(distance / s0 / dt))

@gif for i in 1:NI
  #@timeit to "u .=" u .= A * u
  @timeit to "u .=" u .= A * u
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
