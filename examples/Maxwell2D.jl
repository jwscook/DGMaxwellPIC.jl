using DGMaxwellPIC, Plots, TimerOutputs, StaticArrays, LinearAlgebra, LoopVectorization, SuiteSparse, IterativeSolvers

const NX = 32;
const NY = 8;

const OX = 4;
const OY = 5;

const state2D = State([OX, OY], LegendreNodes);

const DIMS = 2
const L = [NX, NY] .* 2 .* rand(2);

const a = zeros(DIMS);# randn(DIMS);
const b = ones(DIMS) .* L;#a .+ rand(DIMS) .* 10;

gridposition(x) = SVector{DIMS, Float64}((x .* (b .- a) .+ a))

const lowers = [gridposition(((i-1)/NX, (j-1)/NY)) for i in 1:NX, j in 1:NY];
const uppers = [gridposition((i/NX, j/NY)) for i in 1:NX, j in 1:NY];
const grid2D = Grid([Cell(deepcopy(state2D), lowers[i,j], uppers[i,j]) for i in 1:NX, j in 1:NY]);

const centres = [(lowers[i, j] + uppers[i, j])/2 for i in 1:NX, j in 1:NY]
const s0 = DGMaxwellPIC.speedoflight
const k = 4pi / (b[1] - a[1])
const ω = s0 * k

fB(x, t=0) = sin(k * x[1] - ω * t)
fE(x, t=0) = s0 * fB(x, t)

electricfield!(grid2D, fE, 2);
magneticfield!(grid2D, fB, 3);

const dtc = norm((b .- a)./sqrt((NX * OX)^2 + (NY * OY)^2)) / DGMaxwellPIC.speedoflight
const dt = dtc * 1.0
const upwind = 1

@show "Assembling"
const M = assemble(grid2D, upwind=upwind);

#@show "Building Crank-Nicolson"
#const A = (I - Matrix(M) * dt / 2) \ Matrix(I + Matrix(M) * dt / 2);
const CN⁻ = I - M * dt / 2;
const CN⁺ = I + M * dt / 2;
#const luCN⁻ = lu(CN⁻);
@show "Fetching dofs vector"
const u = deepcopy(dofs(grid2D));
const k1 = deepcopy(u);
const k2 = deepcopy(u);
const k3 = deepcopy(u);
const k4 = deepcopy(u);
const work = deepcopy(u);

const ngifevery = 8
const nturns = 1
const NI = Int(ceil(nturns * L[1] / s0 / dt))
const to = TimerOutput()

function myaxpy!(w, a, x, y)
  @tturbo for i in eachindex(y)
    w[i] = a * x[i] + y[i]
  end
end
function rksubstep!(y, w, A, u, k, a)
  myaxpy!(w, a, k, u)
  mul!(y, A, w)
end

function stepfieldsHeun!(u, M, k1, k2, work, dt)
  @timeit to "k1 =" mul!(k1, M, u)
  @timeit to "work =" myaxpy!(work, dt, k1, u)
  @timeit to "k2 =" mul!(k2, M, work)
  @timeit to "u .+= " @tturbo for i in eachindex(u); u[i] += dt * (k1[i] + k2[i]) / 2; end
end

function stepfieldsRK4!(u, M, k1, k2, k3, k4, work, dt)
  @timeit to "k1 =" mul!(k1, M, u)
  @timeit to "k2 =" rksubstep!(k2, work, M, u, k1, dt/2)
  @timeit to "k3 =" rksubstep!(k3, work, M, u, k2, dt/2)
  @timeit to "k4 =" rksubstep!(k4, work, M, u, k3, dt)
  @timeit to "u .+= " @tturbo for i in eachindex(u); u[i] += dt * (k1[i] + 2k2[i] + 2k3[i] + k4[i]) / 6; end
end

@gif for i in 0:NI-1
  #@timeit to "stepfieldsRK4" stepfieldsRK4!(u, M, k1, k2, k3, k4, work, dt)
  #@timeit to "stepfieldsHuen" stepfieldsHeun!(u, M, k1, k2, work, dt)
  #@timeit to "u .= A * u" u .= A * u
  #@timeit to "u .= lu(CN⁻) * CN⁺ * u" SuiteSparse.UMFPACK.solve!(u, luCN⁻, mul!(work, CN⁺, z), 0)
  @timeit to "u .= bigcgstabl! CN" bicgstabl!(u, CN⁻, mul!(work, CN⁺, u), reltol=1000eps())
  if i % ngifevery == 0 # only do this if we need to make plots
    @timeit to "dofs!" dofs!(grid2D, u)
  end
  Eyresult = [electricfield(grid2D, xi, 2) for xi in centres]
  Eexpected = Matrix([fE(centres[ii,jj], i * dt) for ii in 1:NX, jj in 1:NY])
  Bexpected = Matrix([fB(centres[ii,jj], i * dt) for ii in 1:NX, jj in 1:NY])
  err = Float16(norm(Eyresult .- Eexpected) / norm(Eexpected))
  p1 = heatmap(electricfield(grid2D, 1))
  p2 = heatmap(electricfield(grid2D, 2)); title!(p2, "$i of $NI, error = $err")
  p3 = heatmap(electricfield(grid2D, 3))
  p4 = heatmap(Eexpected)
  p5 = heatmap(magneticfield(grid2D, 1))
  p6 = heatmap(magneticfield(grid2D, 2))
  p7 = heatmap(magneticfield(grid2D, 3))
  p8 = heatmap(Bexpected)
  plot(p1, p2, p3, p4, p5, p6, p7, p8; layout = (@layout [a b c d; e f g h]))
  @show i
end every ngifevery
show(to)
