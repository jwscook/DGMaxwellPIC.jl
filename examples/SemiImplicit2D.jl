using DGMaxwellPIC, Plots, TimerOutputs, StaticArrays, LoopVectorization, QuadGK
using LinearAlgebra, StatProfilerHTML, IterativeSolvers, Base.Threads, Random, Statistics
using Profile

function foo()
  NX = 64;
  NY = 4;
  
  OX = 5;
  OY = 5;
  
  state2D = State([OX, OY], LegendreNodes);
  
  DIMS = 2
  
  a = zeros(DIMS);# randn(DIMS);
  b = [1.0, 1.0 * NY/NX];#a .+ rand(DIMS) .* 10;
  area = prod(b .- a)
  
  grid2D = Grid(state2D, a, b, (NX, NY))
  NP = NX * NY * OX * OY * 32
  
  dataxvw = zeros(DIMS + 3 + 1, NP);
  dataxvw[1:DIMS, :] .= rand(DIMS, NP) .* (b .- a) .+ a
  v0 = 0.1
  dataxvw[DIMS+1, :] .= -v0
  dataxvw[DIMS+1, shuffle(1:NP)[1:NP÷2]] .= v0
  dataxvw[DIMS+2, :] .= randn(NP)
  particledata = DGMaxwellPIC.ParticleData(dataxvw);
  weight = 16 * pi^2 / 3 / length(particledata) * v0^2;
  weight!(particledata, weight)
  
  plasma = Plasma([Species(particledata, charge=1.0, mass=1.0)]);
  sort!(plasma, grid2D) # sort particles by cellid
  plasmafuture = deepcopy(plasma)
  
  A = assemble(grid2D);
  u = dofs(grid2D);
  ufuture = deepcopy(u);
  uguess = deepcopy(u) .+ Inf;
  work = deepcopy(u);
  S = deepcopy(u);
  Sfuture = deepcopy(u);
  
  dl = norm(b .- a) / sqrt((NX * OX)^2 + (NY * OY)^2)
  maxprobableparticlespeed = 4 * v0
  dtimplicit = dl / maxprobableparticlespeed / 10

  CN⁻ = I - A * dtimplicit / 2;
  CN⁺ = I + A * dtimplicit / 2;
  luCN⁻ = lu(CN⁻)
  workmatrix = deepcopy(A);
  
  grid2Dfuture = deepcopy(grid2D)
  sort!(plasma, grid2Dfuture)
 
  function work!(output, mat, vec)
    output .= mat
    for j in axes(output, 2)
      output[j,j] += vec[j]
    end
  end

  to = TimerOutput()
  ngifevery = max(2, Int(ceil((b[1]-a[1])/NX / dtimplicit))) * 8
  nturns = 4.0
  NI = Int(ceil((b[1]-a[1]) * nturns / (dtimplicit * v0)))
  cellsize = (b[1] - a[1]) / NX
  ncellscoveredbyfastest = dtimplicit * maxprobableparticlespeed / cellsize
  nsubsteps = Int(ceil(ncellscoveredbyfastest))
  @show dtimplicit, nturns, ngifevery, NI, dtimplicit * NI, nsubsteps, ncellscoveredbyfastest

  t1 =@elapsed @timeit to "source" sources!(S, grid2D)
  @gif for i in 0:NI-1
    t1 += @elapsed begin
      ufuture .= u
      j = 0
      while (j += 1) < 32 && !isapprox(ufuture, uguess, rtol=1e-8, atol=eps())
        uguess .= ufuture
        copyto!(plasmafuture, plasma)
        #=
        @timeit to "stepfields" work!(workmatrix, CN⁺, S)
        @timeit to "stepfields" mul!(work, workmatrix, u)
        @timeit to "stepfields" work!(workmatrix, CN⁻, Sfuture)
        @timeit to "stepfields" bicgstabl!(ufuture, workmatrix, work)
        =#
        @timeit to "stepfields" mul!(work, CN⁺, u)
        @timeit to "stepfields" @tturbo @. work += (S + Sfuture) * dtimplicit / 2
        @timeit to "stepfields" ufuture .= luCN⁻ \ work
        @timeit to "dofs!" dofs!(grid2Dfuture, ufuture)
        for k in 1:nsubsteps
          @timeit to "advance!" advance!(plasmafuture, grid2D, dtimplicit / nsubsteps,
                                         grid2Dfuture, 0.5)#(k-0.5)/nsubsteps)
        end
        @timeit to "deposit" depositcurrent!(grid2Dfuture, plasmafuture)
        @timeit to "source" sources!(Sfuture, grid2Dfuture)
      end
      S .= Sfuture
      u .= ufuture
      copyto!(plasma, plasmafuture)
      @timeit to "dofs!" dofs!(grid2D, u) # not necessary?
      if i % ngifevery == 0 # only do this if we need to make plots
         #current = quadgk(x->currentfield(grid2Dfuture, [x], 1), a[1], b[1])[1]
         #meanfieldcurrentdensity = current / prod(b - a)
         #meanparticlecurrentdensity = sum(DGMaxwellPIC.xvelocity(plasma.species[1])) *
         #  weight / prod(b - a)
         #currentisconsistent = isapprox(meanfieldcurrentdensity, meanparticlecurrentdensity, atol=1e-8)
         x = DGMaxwellPIC.position(plasma.species[1])
         v = DGMaxwellPIC.velocity(plasma.species[1])
         p1 = scatter(x[1, :], v[1, :]); title!("$i of $NI")
         plot(p1)
         #x = collect(1/NX/2:1/NX:1-1/NX/2)
         #Ex = electricfield(grid2D, 1)
         #@show i, extrema(Ex)#, currentisconsistent
         #plot!(x, Ex, ylims=[-1,1])
        #p1 = heatmap(electricfield(grid2D, 1))
        #p2 = heatmap(electricfield(grid2D, 2))
        #p3 = heatmap(electricfield(grid2D, 3))
        #p4 = heatmap(magneticfield(grid2D, 1))
        #p5 = heatmap(magneticfield(grid2D, 2))
        #p6 = heatmap(magneticfield(grid2D, 3))
        #plot(p1, p2, p3, p4, p5, p6, layout = (@layout [a b c; d e f]))
        println(i, ", ", NI, ", ", t1 * (NI / i - 1) / 60, ", ", t1)
      end
      fill!(uguess, Inf)
    end # t1 += @elapsed begin
  end every ngifevery
  show(to)
end

foo()
