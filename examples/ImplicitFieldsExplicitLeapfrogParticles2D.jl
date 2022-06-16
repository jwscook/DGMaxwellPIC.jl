using DGMaxwellPIC, Plots, TimerOutputs, StaticArrays, LoopVectorization, QuadGK
using LinearAlgebra, StatProfilerHTML, IterativeSolvers, Base.Threads, Random, Statistics
using Profile

function foo()
  NX = 64;
  NY = 2;
  
  OX = 5;
  OY = 5;
  
  state2D = State([OX, OY], LegendreNodes);
  
  DIMS = 2
  
  a = zeros(DIMS);
  b = [1.0, NY / NX]
  area = prod(b .- a)
  
  grid2D = Grid(state2D, a, b, (NX, NY))
  NP = NX * NY * OX * OY * 4
  
  dataxvw = zeros(DIMS + 3 + 1, NP);
  dataxvw[1:DIMS, :] .= rand(DIMS, NP) .* (b .- a) .+ a
  v0 = 0.1
  dataxvw[DIMS+1, :] .= -v0
  dataxvw[DIMS+1, shuffle(1:NP)[1:NP÷2]] .= v0
  particledata = DGMaxwellPIC.ParticleData(dataxvw);
  weight = 16 * pi^2 / 3 * area / length(particledata) * v0^2;
  weight!(particledata, weight)
  
  plasma = Plasma([Species(particledata, charge=1.0, mass=1.0)]);
  sort!(plasma, grid2D) # sort particles by cellid
  
  A = assemble(grid2D);
  u = dofs(grid2D);
  work = deepcopy(u);
  S = deepcopy(u);
  
  dl = norm(b .- a) / (NX * OX)
  maxprobableparticlespeed = 3 * v0
  dtimplicit = dl / maxprobableparticlespeed / 10

  CN⁻ = I - A * dtimplicit / 2;
  CN⁺ = I + A * dtimplicit / 2;
  luCN⁻ = lu(CN⁻)
  workmatrix = deepcopy(A);
 
  function work!(output, mat, vec)
    output .= mat
    for j in axes(output, 2)
      output[j,j] += vec[j]
    end
  end

  to = TimerOutput()
  ngifevery = max(2, Int(ceil((b[1]-a[1])/NX / dtimplicit))) * 8
  nturns = 2.0
  NI = Int(ceil((b[1]-a[1]) * nturns / (dtimplicit * v0)))
  cellsize = (b[1] - a[1]) / NX
  ncellscoveredbyfastest = dtimplicit * maxprobableparticlespeed / cellsize
  nsubsteps = Int(ceil(ncellscoveredbyfastest))
  @show dtimplicit, nturns, ngifevery, NI, dtimplicit * NI, nsubsteps, ncellscoveredbyfastest

  @timeit to "source" sources!(S, grid2D)
  @gif for i in 0:NI-1
    #=
    @timeit to "stepfields" work!(workmatrix, CN⁺, S)
    @timeit to "stepfields" mul!(work, workmatrix, u)
    @timeit to "stepfields" work!(workmatrix, CN⁻, S)
    @timeit to "stepfields" bicgstabl!(u, workmatrix, work)
    =#
    for _ in 1:nsubsteps # plasma starts n-1/2 so advance to n+1/2
      @timeit to "advance!" advance!(plasma, grid2D, dtimplicit / nsubsteps)
    end
    @timeit to "deposit" depositcurrent!(grid2D, plasma; zerocurrentfirst=true)
    @timeit to "source" sources!(S, grid2D) # S n+1/2
    @timeit to "stepfields" mul!(work, CN⁺, u) # do implicit timestep from n to n+1
    @timeit to "stepfields" @tturbo @. work += S * dtimplicit
    @timeit to "stepfields" ldiv!(u, luCN⁻, work)
    @timeit to "dofs!" dofs!(grid2D, u) # update grid to n+1
    if i % ngifevery == 0 # only do this if we need to make plots
       #current = quadgk(x->currentfield(grid2D, [x], 1), a[1], b[1])[1]
       #meanfieldcurrentdensity = current / prod(b - a)
       #meanparticlecurrentdensity = sum(DGMaxwellPIC.xvelocity(plasma.species[1])) *
       #  weight / prod(b - a)
       #currentisconsistent = isapprox(meanfieldcurrentdensity, meanparticlecurrentdensity, atol=1e-8)
       x = DGMaxwellPIC.position(plasma.species[1])
       v = DGMaxwellPIC.velocity(plasma.species[1])
       p1 = scatter(x[1, :], v[1, :]); title!("$i of $NI")
       plot(p1)
      #p1 = heatmap(electricfield(grid2D, 1))
      #p2 = heatmap(electricfield(grid2D, 2))
      #p3 = heatmap(electricfield(grid2D, 3))
      #p4 = heatmap(magneticfield(grid2D, 1))
      #p5 = heatmap(magneticfield(grid2D, 2))
      #p6 = heatmap(magneticfield(grid2D, 3))
      #plot(p1, p2, p3, p4, p5, p6, layout = (@layout [a b c; d e f]))
      @show i
    end
  end every ngifevery
  show(to)
end

foo()
