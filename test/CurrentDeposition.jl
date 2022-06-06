using DGMaxwellPIC, Plots, TimerOutputs, StaticArrays, LoopVectorization, StatProfilerHTML
using LinearAlgebra, Random, Statistics, QuadGK, Test

@testset "Current Deposition" begin
  NX = 1;
  
  OX = 9;
  
  state1D = State([OX], LegendreNodes);
  
  DIMS = 1
  
  a = zeros(DIMS);
  b = ones(DIMS) .* rand(DIMS);
  
  gridposition(x) = SVector{DIMS, Float64}((x .* (b .- a) .+ a))
  
  grid1D = Grid([Cell(deepcopy(state1D), gridposition(((i-1)/NX,)), gridposition((i/NX,))) for i in 1:NX]);
  
  NP = NX * OX * 16000
  
  dataxvw = zeros(DIMS + 3 + 1, NP);
  dataxvw[1:DIMS, :] .= rand(DIMS, NP) .* (b .- a) .+ a
  v0 = 5.0 # some number, may as well be prime
  dataxvw[DIMS+1, :] .= v0
  dataxvw[DIMS+1, shuffle(1:NP)[1:NP÷2]] .= v0
  particledata = DGMaxwellPIC.ParticleData(dataxvw);
  n0 = 7.0 # ditto
  nreal = n0 * prod(b .- a)
  W = nreal / length(particledata)
  weight!(particledata, W);
  q0 = 11.0 # ditto

  expectedcurrentdensity = n0 * q0 * v0
 
  plasma = Plasma([Species(particledata, charge=q0, mass=1.0)]);
  sort!(plasma, grid1D) # sort particles by cellid
  totalweight = sum(DGMaxwellPIC.weight(plasma.species[1]))
  S = deepcopy(dofs(grid1D));
 
  depositcurrent!(grid1D, plasma)
  sources!(S, grid1D) # sources known at middle of timestep n+1/2
  currentdofsx = zeros(OX);
  currentdofsx = DGMaxwellPIC.currentfielddofs(grid1D[1].state, 1)

  @test currentfield(grid1D, [0.25] .* prod(b - a), 1) / expectedcurrentdensity ≈ 1 rtol=1e-2
  @test currentfield(grid1D, [0.5] .* prod(b - a), 1) / expectedcurrentdensity ≈ 1 rtol=1e-2
  @test currentfield(grid1D, [0.75] .* prod(b - a), 1) / expectedcurrentdensity ≈ 1 rtol=1e-2
  current = quadgk(x->currentfield(grid1D, [x], 1), 0.0, prod(b-a))[1]
  resultcurrentdensity = current / prod(b - a)
  @test resultcurrentdensity ≈ expectedcurrentdensity
end

