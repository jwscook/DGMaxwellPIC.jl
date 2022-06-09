using DGMaxwellPIC, Plots, TimerOutputs, StaticArrays, LoopVectorization, StatProfilerHTML

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
  NP = NX * NY * OX * OY * 32
  
  dataxvw = zeros(DIMS + 3 + 1, NP);
  dataxvw[1:DIMS, :] .= rand(DIMS, NP) .* (b .- a) .+ a
  dataxvw[DIMS+1, :] .= rand((-1, 1), NP)
  particledata = DGMaxwellPIC.ParticleData(dataxvw);
  weight!(particledata, 32 * pi^2 / 2 * area / length(particledata));
  
  plasma = Plasma([Species(particledata, charge=1.0, mass=1.0)]);
  sort!(plasma, grid2D) # sort particles by cellid
  
  A = assemble(grid2D);
  u = dofs(grid2D);
  k1 = deepcopy(u);
  k2 = deepcopy(u);
  k3 = deepcopy(u);
  k4 = deepcopy(u);
  work = deepcopy(u);
  S = deepcopy(u);
  
  dtc = norm((b .- a)./sqrt((NX * OX)^2 + (NY * OY)^2)) / DGMaxwellPIC.speedoflight
  dt = dtc * 0.1
  
  CN⁻ = I - A * dt / 2;
  CN⁺ = I + A * dt / 2;

  grid2Dcopy = deepcopy(grid2D)
  sort!(plasma, grid2Dcopy)
 
  function myaxpy!(w, a, x, y)
    @tturbo for i in eachindex(y)
      w[i] = a * x[i] + y[i] 
    end
  end
  function rksubstep!(y, w, A, u, k, a, s)
    myaxpy!(w, a, k, u)
    mul!(y, A, w)
    @tturbo for i in eachindex(y); y[i] += s[i]; end
  end
  
  function stepfieldsHeun!(u, M, k1, k2, work, dt, S)
    @timeit to "k1 =" rksubstep!(k1, work, M, u, k1, 0, S)
    @timeit to "k2 =" rksubstep!(k2, work, M, u, k1, dt, S)
    @timeit to "u .+= " @tturbo for i in eachindex(u); u[i] += dt * (k1[i] + k2[i]) / 2; end
  end

  function stepfieldsRK4!(u, M, k1, k2, k3, k4, work, dt, S)
    @timeit to "k1 =" rksubstep!(k1, work, M, u, k1, 0, S)
    @timeit to "k2 =" rksubstep!(k2, work, M, u, k1, dt/2, S)
    @timeit to "k3 =" rksubstep!(k3, work, M, u, k2, dt/2, S)
    @timeit to "k4 =" rksubstep!(k4, work, M, u, k3, dt, S)
    @timeit to "u .+= ..." @tturbo for i in eachindex(u); u[i] += dt * (k1[i] + 2k2[i] + 2k3[i] + k4[i]) / 6; end
  end

  to = TimerOutput()
  ngifevery = Int(ceil((b[1]-a[1])/8NX / dt * v0))
  nturns = 0.5
  NI = Int(ceil((b[1]-a[1]) / dt * nturns * v0))
  @show nturns, ngifevery, NI
  @gif for i in 0:NI-1
    @timeit to "advance!" advance!(plasma, grid2D, dt) # pretend plasma is 1/2 timestep behind fields, so leapfrog to n+1/2
    @timeit to "deposit" depositcurrent!(grid2D, plasma)
    @timeit to "source" sources!(S, grid2D) # sources known at middle of timestep n+1/2
    @timeit to "stepfields" stepfieldsRK4!(u, A, k1, k2, k3, k4, work, dt, S) # advance fields to end of timestep n+1
#    @timeit to "stepfields" bicgstabl!(u, CN⁻, mul!(work, CN⁺, u), reltol=1000eps())
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
     if i % ngifevery == 0 # only do this if we need to make plots
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
