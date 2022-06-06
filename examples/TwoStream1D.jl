using DGMaxwellPIC, Plots, TimerOutputs, StaticArrays, LoopVectorization, StatProfilerHTML
using LinearAlgebra, Random, Statistics

function foo()
  NX = 64;
  
  OX = 5;
  
  state1D = State([OX], LegendreNodes);
  
  DIMS = 1
  
  a = zeros(DIMS);# randn(DIMS);
  b = ones(DIMS);#a .+ rand(DIMS) .* 10;
  area = prod(b .- a)
  
  gridposition(x) = SVector{DIMS, Float64}((x .* (b .- a) .+ a))
  
  grid1D = Grid([Cell(deepcopy(state1D), gridposition(((i-1)/NX,)), gridposition((i/NX,))) for i in 1:NX]);
  
  NP = NX * OX * 32
  
  dataxvw = zeros(DIMS + 3 + 1, NP);
  dataxvw[1:DIMS, :] .= rand(DIMS, NP) .* (b .- a) .+ a
  v0 = 0.1
  dataxvw[DIMS+1, :] .= -v0
  dataxvw[DIMS+1, shuffle(1:NP)[1:NP÷2]] .= v0
  particledata = DGMaxwellPIC.ParticleData(dataxvw);
  weight!(particledata, 32 * pi^2 / 2 * area / length(particledata) * 10);
  
  plasma = Plasma([Species(particledata, charge=1.0, mass=1.0)]);
  sort!(plasma, grid1D) # sort particles by cellid
  
  A = assemble(grid1D);
  u = dofs(grid1D);
  k1 = deepcopy(u);
  k2 = deepcopy(u);
  k3 = deepcopy(u);
  k4 = deepcopy(u);
  work = deepcopy(u);
  S = deepcopy(u);
  
  dtc = norm((b .- a)./(NX * OX)) / DGMaxwellPIC.speedoflight
  dt = dtc * 0.1
  
  CN⁻ = I - A * dt / 2;
  CN⁺ = I + A * dt / 2;

  grid1Dcopy = deepcopy(grid1D)
  sort!(plasma, grid1Dcopy)
 
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
    @timeit to "u .+= ..." @tturbo u += dt * (k1 + 2k2 + 2k3 + k4) / 6
  end

  to = TimerOutput()
  ngifevery = Int(ceil((b[1]-a[1])/8NX / dt)) * 4
  nturns = 2.0
  NI = Int(ceil((b[1]-a[1]) / dt * nturns / v0))
  @show dt, NX, OX, nturns, ngifevery, NI, mean(S), std(S)
  @gif for i in 1:NI
    @timeit to "advance!" advance!(plasma, grid1D, dt) # pretend plasma is 1/2 timestep behind fields, so leapfrog to n+1/2
    @timeit to "deposit" depositcurrent!(grid1D, plasma)
    @timeit to "source" sources!(S, grid1D) # sources known at middle of timestep n+1/2
#    @timeit to "stepfields" stepfieldsHeun!(u, A, k1, k2, work, dt, S) # advance fields to end of timestep n+1
    @timeit to "stepfields" stepfieldsEuler!(u, A, k1, work, dt, S) # advance fields to end of timestep n+1
    #@timeit to "stepfields" stepfieldsRK4!(u, A, k1, k2, k3, k4, work, dt, S) # advance fields to end of timestep n+1
#    @timeit to "stepfields" bicgstabl!(u, CN⁻, mul!(work, CN⁺, u), reltol=1000eps())
    @timeit to "dofs!" dofs!(grid1D, u)

     if i % ngifevery == 1 # only do this if we need to make plots
       x = DGMaxwellPIC.position(plasma.species[1])
       v = DGMaxwellPIC.velocity(plasma.species[1])
       p1 = scatter(x[1, :], v[1, :]); title!("$i of $NI")
       x = collect(1/NX/2:1/NX:1-1/NX/2)
       plot!(x, electricfield(grid1D, 1), ylims=[-1,1])
      #p2 = heatmap(electricfield(grid1D, 2))
      #p3 = heatmap(electricfield(grid1D, 3))
      #p4 = heatmap(magneticfield(grid1D, 1))
      #p5 = heatmap(magneticfield(grid1D, 2))
      #p6 = heatmap(magneticfield(grid1D, 3))
      #plot(p1, p2, p3, p4, p5, p6, layout = (@layout [a b c; d e f]))
      @show i, mean(S) / maximum(abs, S)
    end
  end every ngifevery
  show(to)
end

foo()
