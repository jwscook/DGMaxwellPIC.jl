using DGMaxwellPIC, Plots, TimerOutputs, StaticArrays, LoopVectorization

const NX = 8;
const NY = 4;

const OX = 3;
const OY = 5;

const state2D = State([OX, OY], LegendreNodes);

const DIMS = 2

const a = zeros(DIMS);# randn(DIMS);
const b = ones(DIMS);#a .+ rand(DIMS) .* 10;
const area = prod(b .- a)

gridposition(x) = SVector{DIMS, Float64}((x .* (b .- a) .+ a))

const grid2D = Grid([Cell(deepcopy(state2D), gridposition(((i-1)/NX, (j-1)/NY)), gridposition((i/NX, j/NY))) for i in 1:NX, j in 1:NY]);

const B = 1.0
magneticfield!(grid2D, x->B, 3)

const NP = 100 #NX * NY * OX * OY * 1

const dataxvw = zeros(DIMS + 3 + 1, NP);
dataxvw[1:DIMS, :] .= 0.5 # rand(DIMS, NP) .* (b .- a) .+ a # 0.5
#dataxvw[DIMS+1, :] .= randn(NP) / 3
#dataxvw[DIMS+2, :] .= randn(NP) / 3
θ = rand(NP) .* 2pi .- pi
dataxvw[DIMS+1, :] .= cos.(θ) / 10
dataxvw[DIMS+2, :] .= sin.(θ) / 10
dataxvw[DIMS+3, :] .= randn(NP) / 3
const particledata = DGMaxwellPIC.ParticleData(dataxvw);
weight!(particledata, 32pi * area / length(particledata));
const charge = 1.0
const mass = 1.0
const species1 = Species(particledata, charge=charge, mass=mass);
const plasma = Plasma([species1]);
sort!(plasma, grid2D) # sort particles by cellid

const tau = 2pi / (charge * B / mass)
const dt = tau / 128

sort!(plasma, grid2D)
const species1_ic = deepcopy(species1)

const to = TimerOutput()
const energy0 = sum(abs2, DGMaxwellPIC.velocity(species1))
const nturns = 2
const NI = Int(round(tau/dt*nturns))
@gif for i in eachindex(0:NI)
  @timeit to "advance!" advance!(plasma, grid2D, dt)
  x1 = DGMaxwellPIC.position(species1)
  v1 = DGMaxwellPIC.velocity(species1)
  id1 = DGMaxwellPIC.ids(species1)
  energy = (sum(abs2, v1) - energy0) / energy0
  p1 = scatter(x1[1, :], x1[2, :], c=id1, xlims=[0, 1], ylims=[0,1]); title!("$i of $NI")
  p2 = scatter(v1[1, :], v1[2, :], c=id1, xlims=[-1, 1], ylims=[-1, 1])
  plot(p1, p2, layout = (@layout [a b]))
  @show i, energy
end every 1
show(to)

ids1 = DGMaxwellPIC.ids(species1_ic)
ids2 = DGMaxwellPIC.ids(species1)
x1 = DGMaxwellPIC.position(species1_ic)
v1 = DGMaxwellPIC.velocity(species1_ic)
x2 = DGMaxwellPIC.position(species1)
v2 = DGMaxwellPIC.velocity(species1)
p1 = scatter(x2[1, ids2] .- x1[1, ids1], x2[2, ids2] .- x1[2, ids1])
p2 = scatter(v2[1, ids2] .- v1[1, ids1], v2[2, ids2] .- v1[2, ids1])
plot(p1, p2, layout = (@layout [a b]))
savefig("Boris.png")
