using DGMaxwellPIC, Plots, TimerOutputs, StaticArrays, LoopVectorization, StatProfilerHTML, LinearAlgebra, Random
using Dates, IterativeSolvers, Base.Threads

using ThreadedSparseCSR
ThreadedSparseCSR.multithread_matmul(BaseThreads())


function foo()
  NX = 64;
  NY = 4;
  
  OX = 5;
  OY = 5;
  
  state2D = State([OX, OY], LegendreNodes);
  
  DIMS = 2
  
  a = zeros(DIMS);
  b = [1.0, 1.0 * NY / NX]#ones(DIMS);
  area = prod(b .- a)

  grid2D = Grid(state2D, a, b, (NX, NY))

  #function distributionfunction(xv)
  #  x = xv[1:2]
  #  v = xv[3:5]
  #  return exp(-sum(v.^2))
  #end
  NP = NX * NY * OX * OY * 32
  
  dataxvw = zeros(DIMS + 3 + 1, NP);
  dataxvw[1:DIMS, :] .= rand(DIMS, NP) .* (b .- a) .+ a
  v0 = 0.1
  dataxvw[DIMS+1, :] .= -v0
  dataxvw[DIMS+1, shuffle(1:NP)[1:NP÷2]] .= v0
  particledata = DGMaxwellPIC.ParticleData(dataxvw);
  weight = 16 * pi^2 / 3 * area / length(particledata) * v0^2;
  weight!(particledata, weight);
  
  plasma = Plasma([Species(particledata, charge=1.0, mass=1.0)]);
  sort!(plasma, grid2D) # sort particles by cellid
  
  A = assemble(grid2D, upwind=1.0);
  u = dofs(grid2D);
  k1 = deepcopy(u);
  k2 = deepcopy(u);
  k3 = deepcopy(u);
  k4 = deepcopy(u);
  work = deepcopy(u);
  S = deepcopy(u);
  
  dl = norm((b .- a)./(NX, NY))
  dldg = dl / (OX^2 + OY^2)
  dtc = dldg / DGMaxwellPIC.speedoflight
  maxprobableparticlespeed = 4 * v0
  dtimplicit = dldg / DGMaxwellPIC.speedoflight * 100
  dt = dtimplicit
  
  CN⁻ = I - A * dt / 2;
  CN⁺ = I + A * dt / 2;
  luCN⁻ = lu(CN⁻)
  workmatrix = deepcopy(A);

  grid2Dcopy = deepcopy(grid2D)
  sort!(plasma, grid2Dcopy)
 
  plasmafuture = deepcopy(plasma)
  
  ufuture = deepcopy(u);
  uguess = deepcopy(u) .+ Inf;
  Sfuture = deepcopy(u);
  Smid = deepcopy(u);
 
  grid2Dfuture = deepcopy(grid2D)


  function myaxpy!(w, a, x, y)
    @tturbo for i in eachindex(y)
      w[i] = a * x[i] + y[i] 
    end
  end
  function rksubstep!(y, w, A, u, k, dt, s)
    myaxpy!(w, dt, k, u)
    mul!(y, A, w)
    @tturbo for i in eachindex(y); y[i] += s[i]; end
  end
  
  function stepfieldsEuler!(u, M, k1, work, dt, S)
    @timeit to "k1 =" rksubstep!(k1, work, M, u, k1, 0, S)
    @timeit to "u .+= ..." @tturbo @. u += dt * k1
  end
  function stepfieldsHeun!(u, M, k1, k2, work, dt, S)
    @timeit to "k1 =" rksubstep!(k1, work, M, u, k1, 0dt, S)
    @timeit to "k2 =" rksubstep!(k2, work, M, u, k1, dt, S)
    @timeit to "u .+= " @tturbo @. u += dt * (k1 + k2) / 2
  end

  function stepfieldsRK4!(u, M, k1, k2, k3, k4, work, dt, S)
    @timeit to "k1 =" rksubstep!(k1, work, M, u, k1, 0dt, S)
    @timeit to "k2 =" rksubstep!(k2, work, M, u, k1, dt/2, S)
    @timeit to "k3 =" rksubstep!(k3, work, M, u, k2, dt/2, S)
    @timeit to "k4 =" rksubstep!(k4, work, M, u, k3, dt, S)
    @timeit to "u .+= ..." @tturbo @. u += dt * (k1 + 2k2 + 2k3 + k4) / 6
  end

  to = TimerOutput()
  ngifevery = Int(ceil((b[1]-a[1])/8NX / dt)) * 16
  nturns = 2.0
  NI = 640 # Int(ceil((b[1]-a[1]) * nturns  / (dt * v0)))
  nsubsteps = 1
  @show nturns, ngifevery, NI
  @timeit to "source" sources!(S, grid2D) # sources known at middle of timestep n+1/2
  t1 = 0.0
  @gif for i in 0:NI-1
    t1 += @elapsed begin
      ufuture .= u
      j = 0
      while (j += 1) < 32 && !isapprox(ufuture, uguess, rtol=1e-12, atol=eps()^2)
        uguess .= ufuture
        @timeit to "copyto! plasma" copyto!(plasmafuture, plasma) # reset plasmafuture to the start
        for k in 1:nsubsteps
          @timeit to "advance!" advance!(plasmafuture, grid2D, dt / nsubsteps,
                                         grid2Dfuture, (k-0.5)/nsubsteps)
        end

        @timeit to "deposit" depositcurrent!(grid2Dfuture, plasmafuture)
        @timeit to "source" sources!(Sfuture, grid2Dfuture)
        @timeit to "stepfields" begin
          @timeit to "mul!" mul!(work, CN⁺, u)
          @timeit to "@tturbo @. +=" @tturbo @. work += (S + Sfuture) * dt / 2
          @timeit to "bicgstabl!" bicgstabl!(ufuture, CN⁻, work, reltol=1e-8)
        end
        @timeit to "dofs!" dofs!(grid2Dfuture, ufuture)
      end
      @timeit to "newstep copies" begin
        S, Sfuture = Sfuture, S
        u, ufuture = ufuture, u
        plasma, plasmafuture = plasmafuture, plasma
        grid2D, grid2Dfuture = grid2Dfuture, grid2D
        @turbo uguess .+= Inf
      end
      breakout = false

      if i % ngifevery == 0 # only do this if we need to make plots
        x = DGMaxwellPIC.position(plasma.species[1])
        v = DGMaxwellPIC.velocity(plasma.species[1])
        p0 = scatter(x[1, :], v[1, :], xlim=(0, 1), ylim=(-1, 1)); title!("$i of $NI")
        #plot(p0)
        p1 = heatmap(electricfield(grid2D, 1))
        p2 = heatmap(electricfield(grid2D, 2))
        p3 = heatmap(electricfield(grid2D, 3))
        p4 = heatmap(magneticfield(grid2D, 1))
        p5 = heatmap(magneticfield(grid2D, 2))
        p6 = heatmap(magneticfield(grid2D, 3))
        p123 = heatmap(DGMaxwellPIC.divE(grid2D))
        p456 = heatmap(DGMaxwellPIC.divB(grid2D))
        plot(p0, p1, p2, p3, p123, p4, p5, p6, p456, layout = (@layout [a; b c d e; f g h i]))
        breakout = any(vi->abs(vi)>1, v)
        println(i, " of ",NI, ": $(Int(round(100i/NI)))%, approx time left ", Time(0) + Second(round(t1 * (NI / (i + 1) - 1))), ", particle subits = ", j)
      end
      breakout && break
    end # elapsed
  end every ngifevery
  show(to)
  end

foo()
