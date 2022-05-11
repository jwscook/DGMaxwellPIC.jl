using DGMaxwellPIC, Plots, TimerOutputs, StaticArrays, LoopVectorization, StatProfilerHTML

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

#function distributionfunction(xv)
#  x = xv[1:2]
#  v = xv[3:5]
#  return exp(-sum(v.^2))
#end
const NP = NX * NY * OX * OY * 32

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
const k1 = deepcopy(u);
const k2 = deepcopy(u);
const k3 = deepcopy(u);
const k4 = deepcopy(u);
const work = deepcopy(u);

const dtc = norm((b .- a)./sqrt((NX * OX)^2 + (NY * OY)^2)) / DGMaxwellPIC.speedoflight
const dt = dtc *0.1

const grid2Dcopy = deepcopy(grid2D)
sort!(plasma, grid2Dcopy)

function substep!(y, w, A, u, k, a)
  @tturbo for i in eachindex(y)
    w[i] = u[i] + k[i] * a
  end
  mul!(y, A, w)
end

const to = TimerOutput()
function stepfields!(u, M, k1, k2, k3, k4, work, dt, S)
  @timeit to "k1 =" mul!(k1, M, u)
  @timeit to "k2 =" substep!(k2, work, M, u, k1, dt/2)
  @timeit to "k3 =" substep!(k3, work, M, u, k2, dt/2)
  @timeit to "k4 =" substep!(k4, work, M, u, k3, dt)
  @timeit to "u .+= RK4" @tturbo for i in eachindex(u); u[i] += dt * (k1[i] + 2k2[i] + 2k3[i] + k4[i]) / 6; end
  @timeit to "u .+= S" @tturbo for i in eachindex(u); u[i] += S[i]; end
end

@gif for i in 1:64
  @timeit to "advance!" advance!(plasma, grid2D, dt)
  @timeit to "deposit" depositcurrent!(grid2D, plasma)
  @timeit to "source" S .= sources(grid2D)
  @timeit to "RK4" stepfields!(u, A, k1, k2, k3, k4, work, dt, S)
  @timeit to "dofs!" dofs!(grid2D, u)
  if i == 1
    @profilehtml begin
      advance!(plasma, grid2D, 0.0)
      depositcurrent!(grid2D, plasma)
      S .= sources(grid2D)
      stepfields!(u, A, k1, k2, k3, k4, work, 0.0, S)
      dofs!(grid2D, u)
      advance!(plasma, grid2D, 0.0)
    end
  end
  p1 = heatmap(electricfield(grid2D, 1))
  p2 = heatmap(electricfield(grid2D, 2))
  p3 = heatmap(electricfield(grid2D, 3))
  p4 = heatmap(magneticfield(grid2D, 1))
  p5 = heatmap(magneticfield(grid2D, 2))
  p6 = heatmap(magneticfield(grid2D, 3))
  plot(p1, p2, p3, p4, p5, p6, layout = (@layout [a b c; d e f]))
  @show i
end every 4
show(to)
