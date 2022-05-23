using DGMaxwellPIC, Plots, TimerOutputs, StaticArrays, LoopVectorization, StatProfilerHTML, IterativeSolvers

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
  dataxvw[DIMS+1, :] .= rand((-1, 1), NP)
  particledata = DGMaxwellPIC.ParticleData(dataxvw);
  weight!(particledata, 32 * pi^2 / 2 * area / length(particledata) * 10);

  
  plasma = Plasma([Species(particledata, charge=1.0, mass=1.0)]);
  sort!(plasma, grid2D) # sort particles by cellid
  plasmacopy = deepcopy(plasma)
  
  A = assemble(grid2D);
  u = dofs(grid2D);
  u1 = deepcopy(u);
  work = deepcopy(u);
  S = deepcopy(u);
  S1 = deepcopy(u);
  
  dtc = norm((b .- a)./sqrt((NX * OX)^2 + (NY * OY)^2)) / DGMaxwellPIC.speedoflight
  dtfields = dtc
  dtparticles = norm(b .- a) / sqrt(NX^2 + NY^2)  * 0.2
  nsubsteps = Int(ceil(dtfields / dtparticles)) # find round number
  dtparticles = dtfields / nsubsteps # correct particle substep dt

  CN⁻ = I - A * dtfields / 2;
  CN⁺ = I + A * dtfields / 2;
  workmatrix = deepcopy(A);
  
  grid2Dcopy = deepcopy(grid2D)
  sort!(plasma, grid2Dcopy)
 
  to = TimerOutput()
  ngifevery = Int(ceil((b[1]-a[1])/8NX / dtfields))
  nturns = 4.0
  NI = Int(ceil((b[1]-a[1]) / dtfields * nturns))
  @show nturns, ngifevery, NI
  @gif for i in 1:NI
    u1 .= u
    for _ in 1:5 # convergence
      copyto!(plasma, plasmacopy)
      @timeit to "deposit" depositcurrent!(grid2D, plasma)
      @timeit to "source" (sources!(S, grid2D); S .*= dtfields)
      @timeit to "stepfields" @tturbo for i in eachindex(CN⁺); workmatrix[i] = CN⁺[i]; end
      @timeit to "stepfields" @tturbo for i in 1:size(CN⁺, 1); workmatrix[i, i] += S[i]; end
      @timeit to "stepfields" mul!(work, workmatrix, u)
      @timeit to "stepfields" @tturbo for i in eachindex(CN⁻); workmatrix[i] = CN⁻[i]; end
      @timeit to "stepfields" @tturbo for i in 1:size(CN⁻, 1); workmatrix[i, i] += S1[i]; end
      @timeit to "stepfields" bicgstabl!(u1, CN⁻ + diagm(S1), work)
      @timeit to "stepfields" bicgstabl!(u1, CN⁻ + diagm(S1), mul!(work, (workmatrix .= CN⁺ .+ diagm(S)), u))
      @timeit to "stepfields" bicgstabl!(u1, CN⁻ + diagm(S1), mul!(work, CN⁺ + diagm(S), u))
      @timeit to "dofs!" dofs!(grid2Dcopy, u1)
      for j in 1:nsubsteps
        @timeit to "advance!" advance!(plasma, grid2D, dtfields, grid2Dcopy, j / nsubsteps)
      end
      @timeit to "deposit" depositcurrent!(grid2D, plasma)
      @timeit to "source" (sources!(S1, grid2D); S1 .*= dtfields)
    end
    u .= u1
    copyto!(plasmacopy, plasma)
    @timeit to "dofs!" dofs!(grid2D, u)
#    if i == 1
#      @profilehtml begin
#        for j in 1:10
#          advance!(plasma, grid2D, 0.0)
#          depositcurrent!(grid2D, plasma)
#          sources!(S, grid2D)
#          stepfields!(u, A, k1, k2, k3, k4, work, 0.0, S)
#          dofs!(grid2D, u)
#          advance!(plasma, grid2D, 0.0)
#        end
#      end
#    end
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
