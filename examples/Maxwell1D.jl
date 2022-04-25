using DGMaxwellPIC, Plots, TimerOutputs, StaticArrays, LinearAlgebra

const NX = 32;

const OX = 3;

const state1D = State([OX], LobattoNodes);
#const state1D = State([OX], LegendreNodes);

const DIMS = 1
const L = sqrt(2) * 10 #rand() * 10.0

const a = zeros(DIMS);
const b = ones(DIMS) .* L;
const area = prod(b .- a)

gridposition(x) = SVector{DIMS, Float64}((x .* (b .- a) .+ a))

const grid1D = Grid([Cell(deepcopy(state1D), gridposition((i-1)/NX), gridposition(i/NX)) for i in 1:NX]);

const s0 = DGMaxwellPIC.speedoflight
const k = 4pi / L
const ω = s0 * k

fBz(x, t=0) = sin(k * x[1] - ω * t)
fEy(x, t=0) = s0 * fBz(x, t)

electricfield!(grid1D, fEy, 2);
magneticfield!(grid1D, fBz, 3);
#DGMaxwellPIC.electricfielddofs!(grid1D, s0, 2);
#DGMaxwellPIC.magneticfielddofs!(grid1D, 1.0, 3);

const dtc = minimum((b .- a)./NX./OX) / s0
const dt = dtc * 0.95

# du/dt = A * u
# u1 - u0 = dt * (A * u)
# u1 - u0 = dt * (A * (u1 + u0)/2)
# u1 = u0 + dt/2 * A * u1 + dt/2 * A * u0
# (1 - dt/2 * A)*u1 = (1 + dt/2 * A) * u0
# u1 = (1 - dt/2 * A)^-1 (1 + dt/2 * A) * u0

const C = assemble(grid1D, upwind=0.0) * dt / 2;
const B = lu(I - C);
const A = B \ Matrix(I + C);
#const A = I + assemble(grid1D, upwind=0.0) * dt;
const S = sources(grid1D);
const u = dofs(grid1D);

const to = TimerOutput()
const x = collect(1/NX/2:1/NX:1-1/NX/2) .* L

const nturns = 1
const NI = Int(ceil(nturns * L/ s0 / dt))

@gif for i in 1:NI
  #@timeit to "u .=" u .= A * u
  @timeit to "u .=" u .= A * u
  @timeit to "dofs!" dofs!(grid1D, u)
  t = i * dt
  p1 = plot(x, electricfield(grid1D, 1), ylims=[-s0,s0]); title!("$i of $NI")
  p2 = plot(x, electricfield(grid1D, 2), ylims=[-s0,s0])
  p3 = plot(x, electricfield(grid1D, 3), ylims=[-s0,s0])
  p4 = plot(x, magneticfield(grid1D, 1), ylims=[-1,1])
  p5 = plot(x, magneticfield(grid1D, 2), ylims=[-1,1])
  p6 = plot(x, magneticfield(grid1D, 3), ylims=[-1,1])
  plot!(p2, x, [fEy([xi], t) for xi in x], ylims=[-s0,s0])
  plot!(p6, x, [fBz([xi], t) for xi in x], ylims=[-1,1])
  plot(p1, p2, p3, p4, p5, p6, layout = (@layout [a b c; d e f]))
  @show i, i * dt * s0
end every 1
show(to)
