using DGMaxwellPIC, Plots, TimerOutputs, StaticArrays, LinearAlgebra, LoopVectorization

const NX = 64;

const OX = 5;

#const state1D = State([OX], LobattoNodes);
const state1D = State([OX], LegendreNodes);

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
const dt = dtc * 0.25
const upwind = 1

# du/dt = a * u
# u1 - u0 = dt * (a * u)
# u1 - u0 = dt * (a * (u1 + u0)/2)
# u1 = u0 + dt/2 * a * u1 + dt/2 * a * u0
# (1 - dt/2 * a)*u1 = (1 + dt/2 * a) * u0
# u1 = (1 - dt/2 * a)^-1 (1 + dt/2 * a) * u0

const M = assemble(grid1D, upwind=upwind);
const C = M * dt;
const Acranknicolson = (I - C * 0.5) \ Matrix(I + C * 0.5);

const Aforwardeuler = I + M * dt;
const Aalmostcranknicolson = (I - C * 0.6) \ Matrix(I + C * 0.4);
const Abackwardeuler = inv(Matrix(I - C))

# du/dt = a * u
# u1 - u0 = dt * (a * u)
# u1 - u0 = dt * (a * u1)
# u1 = u0 + dt * a * u1
# (1 - dt * a)*u1 = u0
# u1 = (1 - dt * a)^-1 * u0
#
#const Abackwardeuler = inv(I - C);
#
const A = Acranknicolson
#const A = Aalmostcranknicolson
#const A = Aforwardeuler
#const A = Abackwardeuler

const S = sources(grid1D);
const u = dofs(grid1D);

const to = TimerOutput()
const x = collect(1/NX/2:1/NX:1-1/NX/2) .* L

const nturns = 2
const NI = Int(ceil(nturns * L / s0 / dt))
const k1 = deepcopy(u)
const k2 = deepcopy(u)
const k3 = deepcopy(u)
const k4 = deepcopy(u)
const work = deepcopy(u)

function substep!(y, w, A, u, k, a)
  @tturbo for i in eachindex(y)
    w[i] = u[i] + k[i] * a
  end
  mul!(y, A, w)
end
@gif for i in 1:NI
  @timeit to "k1 =" mul!(k1, M, u)
  @timeit to "k4 =" substep!(k2, work, M, u, k1, dt/2)
  @timeit to "k4 =" substep!(k3, work, M, u, k2, dt/2)
  @timeit to "k4 =" substep!(k4, work, M, u, k2, dt)
  @timeit to "u .+=" @tturbo for i in eachindex(u); u[i] += dt * (k1[i] + 2k2[i] + 2k3[i] + k4[i]) / 6; end
  #@timeit to "u .= A * u" u .= A * u
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
end every 8
show(to)
