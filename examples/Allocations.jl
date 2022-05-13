using DGMaxwellPIC, Plots, TimerOutputs, StaticArrays, LoopVectorization, InteractiveUtils

const NX = 16;
const NY = 8;

const OX = 3;
const OY = 5;

const state2D = State([OX, OY], LegendreNodes);

const DIMS = 2

const a = zeros(DIMS);# randn(DIMS);
const b = ones(DIMS);#a .+ rand(DIMS) .* 10;
const area = prod(b .- a)

gridposition(x) = SVector{DIMS, Float64}((x .* (b .- a) .+ a))

const grid2D = Grid([Cell(deepcopy(state2D), gridposition(((i-1)/NX, (j-1)/NY)), gridposition((i/NX, j/NY))) for i in 1:NX, j in 1:NY]);
const NP = NX * NY * OX * OY * 10

function createparticledata(np)
  dataxvw = zeros(DIMS + 3 + 1, np);
  dataxvw[1:DIMS, :] .= rand(DIMS, np) .* (b .- a) .+ a
  dataxvw[DIMS+1, :] .= rand((-1, 1), np)
  particledata = DGMaxwellPIC.ParticleData(dataxvw);
  weight!(particledata, 32pi * area / length(particledata));
  return particledata
end

const littleplasma = Plasma([Species(createparticledata(1), charge=1.0, mass=1.0)]);
const plasma = Plasma([Species(createparticledata(NP), charge=1.0, mass=1.0)]);
sort!(plasma, grid2D) # sort particles by cellid

const u = dofs(grid2D);
const S = deepcopy(u);
sort!(plasma, grid2D)
#const to = TimerOutput()
#  const dt = 1.0
#  const cellids = DGMaxwellPIC.cellids(littleplasma.species[1])
#  const pos = DGMaxwellPIC.position(littleplasma.species[1])
#  const output = DGMaxwellPIC.workarrays(littleplasma.species[1])[2]
#  DGMaxwellPIC.electromagneticfield!(output, grid2D, cellids, pos)
#  DGMaxwellPIC.electromagneticfield!(output, grid2D, cellids, pos)
#  const dofs1 = zeros(OX, OY)
#  const dofs2 = zeros(OX, OY)
#  const dofs3 = zeros(OX, OY)
#  const dofs4 = zeros(OX, OY)
#  const dofs5 = zeros(OX, OY)
#  const dofs6 = zeros(OX, OY)
#  const state1 = deepcopy(state2D)
#  const output1 = zeros(6)
#  const x1 = rand(2) 
#  const nodes = DGMaxwellPIC.ndimnodes(grid2D, grid2D.data[1])
#  @time DGMaxwellPIC.lagrange!(output1, x1, nodes, (dofs1, dofs2, dofs3, dofs4, dofs5, dofs6)...)
#  @code_warntype DGMaxwellPIC.lagrange!(output1, x1, nodes, (dofs1, dofs2, dofs3, dofs4, dofs5, dofs6)...)
#  @show @time DGMaxwellPIC.lagrange!(output1, x1, nodes, (dofs1, dofs2, dofs3, dofs4, dofs5, dofs6)...)
#  @show @allocated DGMaxwellPIC.lagrange!(output1, x1, nodes, (dofs1, dofs2, dofs3, dofs4, dofs5, dofs6)...)
#  @time DGMaxwellPIC.lagrange(x1, nodes, (1, 1))
#  @show @allocated DGMaxwellPIC.lagrange(x1, nodes, (1, 1))

@time begin
advance!(littleplasma, grid2D, dt)
depositcurrent!(grid2D, littleplasma)
sources!(S, grid2D)
dofs!(grid2D, u)
end

@time begin
advance!(plasma, grid2D, dt)
depositcurrent!(grid2D, plasma)
sources!(S, grid2D)
dofs!(grid2D, u)
end

