using DGMaxwellPIC, Plots, TimerOutputs, StaticArrays, LoopVectorization
using LinearAlgebra, StatProfilerHTML, IterativeSolvers, Base.Threads

function foo()
  NX = 64;
  NY = 4;
  
  OX = 5;
  OY = 5;
  
  state2D = State([OX, OY], LegendreNodes);
  
  DIMS = 2
  
  a = zeros(DIMS);# randn(DIMS);
  b = ones(DIMS);#a .+ rand(DIMS) .* 10;
  area = prod(b .- a)
  
  gridposition(x) = SVector{DIMS, Float64}((x .* (b .- a) .+ a))
  
  grid2D = Grid([Cell(deepcopy(state2D), gridposition(((i-1)/NX, (j-1)/NY)), gridposition((i/NX, j/NY))) for i in 1:NX, j in 1:NY]);

  #function distributionfunction(xv)
  #  x = xv[1:2]
  #  v = xv[3:5]
  #  return exp(-sum(v.^2))
  #end
  NP = NX * NY * OX * OY * 8
  
  dataxvw = zeros(DIMS + 3 + 1, NP);
  dataxvw[1:DIMS, :] .= rand(DIMS, NP) .* (b .- a) .+ a
  v0 = 0.1
  dataxvw[DIMS+1, :] .= rand((-v0, v0), NP)
  particledata = DGMaxwellPIC.ParticleData(dataxvw);
  weight!(particledata, 32 * pi^2 / 2 * area / length(particledata) * 10);

  
  plasma = Plasma([Species(particledata, charge=1.0, mass=1.0)]);
  sort!(plasma, grid2D) # sort particles by cellid
  plasmafuture = deepcopy(plasma)
  
  A = assemble(grid2D);
  u = dofs(grid2D);
  ufuture = deepcopy(u);
  work = deepcopy(u);
  S = deepcopy(u);
  Sfuture = deepcopy(u);
  
  dtc = norm((b .- a)./sqrt((NX * OX)^2 + (NY * OY)^2)) / DGMaxwellPIC.speedoflight
  dtcdg = dtc / 10
  dtfields = dtc
  dl = norm(b .- a) / sqrt(NX^2 + NY^2)
  maxprobableparticlespeed = 3
  dtparticles = dl / maxprobableparticlespeed
  nsubsteps = Int(ceil(dtfields / dtparticles)) # find round number
  dtparticles = dtfields / nsubsteps # correct particle substep dt

  CN⁻ = I - A * dtfields / 2;
  CN⁺ = I + A * dtfields / 2;
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
  ngifevery = max(2, Int(ceil((b[1]-a[1])/8NX / dtfields)))
  nturns = 2.0
  NI = 1000 # Int(ceil((b[1]-a[1]) / dtfields * nturns))

  @timeit to "source" sources!(S, grid2D)
  @show nturns, ngifevery, NI
  @gif for i in 1:NI
    ufuture .= u
    Sfuturenorm = norm(Sfuture)
    copyto!(plasmafuture, plasma)
    normu = Inf
    j = 0
    while j < 10 && !isapprox(norm(ufuture), normu, rtol=1e-3, atol=100eps())
      j += 1
      normu = norm(ufuture)
      #=
      @timeit to "stepfields" work!(workmatrix, CN⁺, S)
      @timeit to "stepfields" mul!(work, workmatrix, u)
      @timeit to "stepfields" work!(workmatrix, CN⁻, Sfuture)
      @timeit to "stepfields" bicgstabl!(ufuture, workmatrix, work)
      =#
      #@show ufuture[1:100], workmatrix[1:100, 1:100], work[1:100]
      #@timeit to "stepfields" ufuture .= CN⁻ \ mul!(work, CN⁺, u) .+ (S .+ Sfuture) .* dtfields / 2
      @timeit to "stepfields" ufuture .= luCN⁻ \ mul!(work, CN⁺, u)
      @timeit to "stepfields" @tturbo @. work += (S + Sfuture) * dtfields / 2
      @timeit to "stepfields" ufuture .+= luCN⁻ \ work
      @timeit to "dofs!" dofs!(grid2Dfuture, ufuture)
      for j in 1:nsubsteps
        @timeit to "advance!" advance!(plasmafuture, grid2D, dtfields, grid2Dfuture, j / nsubsteps)
      end
      @timeit to "deposit" depositcurrent!(grid2Dfuture, plasmafuture)
      @timeit to "source" sources!(Sfuture, grid2Dfuture)
      @show j, 1 - norm(ufuture) / normu
    end
    S .= Sfuture
    u .= ufuture
    copyto!(plasma, plasmafuture)
    @timeit to "dofs!" dofs!(grid2D, u)
    if i % ngifevery == 1 # only do this if we need to make plots
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
    end
    @show i
  end every ngifevery
  show(to)
  end

foo()
