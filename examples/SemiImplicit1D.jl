using DGMaxwellPIC, Plots, TimerOutputs, StaticArrays, LoopVectorization, QuadGK
using LinearAlgebra, StatProfilerHTML, IterativeSolvers, Base.Threads, Random, Statistics
using Profile

function foo()
  NX = 64;
  
  OX = 5;
  
  state1D = State([OX], LegendreNodes);
  
  DIMS = 1
  
  a = zeros(DIMS);# randn(DIMS);
  b = ones(DIMS);#a .+ rand(DIMS) .* 10;
  area = prod(b .- a)
  
  gridposition(x) = SVector{DIMS, Float64}((x .* (b .- a) .+ a))
  
  grid1D = Grid([Cell(deepcopy(state1D), gridposition((i-1)/NX), gridposition(i/NX)) for i in 1:NX]);
  NP = NX * OX * 32
  
  dataxvw = zeros(DIMS + 3 + 1, NP);
  dataxvw[1:DIMS, :] .= rand(DIMS, NP) .* (b .- a) .+ a
  v0 = 0.1
  dataxvw[DIMS+1, :] .= -v0
  dataxvw[DIMS+1, shuffle(1:NP)[1:NP÷2]] .= v0
  particledata = DGMaxwellPIC.ParticleData(dataxvw);
  weight = 16 * pi^2 / 3 / length(particledata) * v0^2;
  weight!(particledata, weight)
  
  plasma = Plasma([Species(particledata, charge=1.0, mass=1.0)]);
  sort!(plasma, grid1D) # sort particles by cellid
  plasmafuture = deepcopy(plasma)
  
  A = assemble(grid1D);
  u = dofs(grid1D);
  ufuture = deepcopy(u);
  work = deepcopy(u);
  S = deepcopy(u);
  Sfuture = deepcopy(u);
  
  dl = norm(b .- a) / (NX * OX)
  maxprobableparticlespeed = 4 * v0
  dtimplicit = dl / maxprobableparticlespeed

  CN⁻ = I - A * dtimplicit / 2;
  CN⁺ = I + A * dtimplicit / 2;
  luCN⁻ = lu(CN⁻)
  workmatrix = deepcopy(A);
  
  grid1Dfuture = deepcopy(grid1D)
  sort!(plasma, grid1Dfuture)
 
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
  cellxingtime = (b[1] - a[1]) / NX / v0
  nsubsteps = Int(ceil(dtimplicit / cellxingtime))
  @show dtimplicit, nturns, ngifevery, NI, dtimplicit * NI, nsubsteps

  @timeit to "source" sources!(S, grid1D)
  @gif for i in 0:NI-1
    ufuture .= u
    Sfuturenorm = norm(Sfuture)
    normu = Inf
    j = 0
    while (j += 1) < 10 && !isapprox(norm(ufuture), normu, rtol=1e-8, atol=100eps())
      normu = norm(ufuture)
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
      @timeit to "dofs!" dofs!(grid1Dfuture, ufuture)
      #@profilehtml for i in 1:100000
      #    dofs!(grid1Dfuture, ufuture)
      #end
      for j in 1:nsubsteps
        @timeit to "advance!" advance!(plasmafuture, grid1D, dtimplicit / nsubsteps,
          grid1Dfuture, (j-0.5)/nsubsteps)
      end

##      @profilehtml begin
##          for j in 1:1000
##              @timeit to "advance!" advance!(plasmafuture, grid1D, dtimplicit / nsubsteps,grid1Dfuture, 0.5)
##          end
##      end
##      throw(error("dasfafjabnfdsflin"))
      @timeit to "deposit" depositcurrent!(grid1Dfuture, plasmafuture)
      @timeit to "source" sources!(Sfuture, grid1Dfuture)
    end
    S .= Sfuture
    u .= ufuture
    copyto!(plasma, plasmafuture)
    @timeit to "dofs!" dofs!(grid1D, u) # not necessary?
    if i % ngifevery == 0 # only do this if we need to make plots
       #current = quadgk(x->currentfield(grid1Dfuture, [x], 1), a[1], b[1])[1]
       #meanfieldcurrentdensity = current / prod(b - a)
       #meanparticlecurrentdensity = sum(DGMaxwellPIC.xvelocity(plasma.species[1])) *
       #  weight / prod(b - a)
       #currentisconsistent = isapprox(meanfieldcurrentdensity, meanparticlecurrentdensity, atol=1e-8)
       x = DGMaxwellPIC.position(plasma.species[1])
       v = DGMaxwellPIC.velocity(plasma.species[1])
       p1 = scatter(x[1, :], v[1, :]); title!("$i of $NI")
       plot(p1)
       x = collect(1/NX/2:1/NX:1-1/NX/2)
       Ex = electricfield(grid1D, 1)
       @show i, extrema(Ex)#, currentisconsistent
       plot!(x, Ex, ylims=[-1,1])
      #p1 = heatmap(electricfield(grid1D, 1))
      #p2 = heatmap(electricfield(grid1D, 2))
      #p3 = heatmap(electricfield(grid1D, 3))
      #p4 = heatmap(magneticfield(grid1D, 1))
      #p5 = heatmap(magneticfield(grid1D, 2))
      #p6 = heatmap(magneticfield(grid1D, 3))
      #plot(p1, p2, p3, p4, p5, p6, layout = (@layout [a b c; d e f]))
    end
  end every ngifevery
  show(to)
end

foo()
