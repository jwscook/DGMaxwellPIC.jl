using DGMaxwellPIC, Plots, TimerOutputs, StaticArrays, LoopVectorization

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

const dataxvw = zeros(DIMS + 3 + 1, NP);
dataxvw[1:DIMS, :] .= rand(DIMS, NP) .* (b .- a) .+ a
dataxvw[DIMS+1, :] .= rand((-1, 1), NP)
const particledata = DGMaxwellPIC.ParticleData(dataxvw);
weight!(particledata, 32pi * area / length(particledata));

const plasma = Plasma([Species(particledata, charge=1.0, mass=1.0)]);
sort!(plasma, grid2D) # sort particles by cellid

const S = sources(grid2D);
const u = dofs(grid2D);
sort!(plasma, grid2D)
const to = TimerOutput()
const dt = 1.0

@time begin
advance!(plasma, grid2D, dt)
depositcurrent!(grid2D, plasma)
S .= sources(grid2D)
dofs!(grid2D, u)
end

@time begin
advance!(plasma, grid2D, dt)
depositcurrent!(grid2D, plasma)
S .= sources(grid2D)
dofs!(grid2D, u)
end

