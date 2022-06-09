using DGMaxwellPIC, Plots, TimerOutputs, StaticArrays, LoopVectorization, StatProfilerHTML
using LinearAlgebra, Random, Statistics, QuadGK

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
  
  NP = NX * OX * 8
  
  dataxvw = zeros(DIMS + 3 + 1, NP);
  dataxvw[1:DIMS, :] .= rand(DIMS, NP) .* (b .- a) .+ a
  v0 = 0.1
  dataxvw[DIMS+1, :] .= -v0
  dataxvw[DIMS+1, shuffle(1:NP)[1:NP÷2]] .= v0
  particledata = DGMaxwellPIC.ParticleData(dataxvw);
  weight = 16 * pi^2 / 3 / length(particledata) * v0^2;
  weight!(particledata, weight);
  
  plasma = Plasma([Species(particledata, charge=1.0, mass=1.0)]);
  sort!(plasma, grid1D) # sort particles by cellid
  
  A = assemble(grid1D, upwind=1.0);
  u = dofs(grid1D);
  k1 = deepcopy(u);
  k2 = deepcopy(u);
  k3 = deepcopy(u);
  k4 = deepcopy(u);
  work = deepcopy(u);
  S = deepcopy(u);
  S1 = deepcopy(u);
  
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
    @timeit to "u .+= ..." @tturbo @. u += dt * (k1 + 2k2 + 2k3 + k4) / 6
  end

  to = TimerOutput()
  ngifevery = Int(ceil((b[1]-a[1])/8NX / dt)) * 16
  nturns = 2.0
  NI = Int(ceil((b[1]-a[1]) * nturns  / (dt * v0)))
  @show dt, NX, OX, nturns, ngifevery, NI, dt * NI
  @timeit to "source" sources!(S1, grid1D) # sources known at middle of timestep n+1/2
  @gif for i in 0:NI-1
    # pretend plasma is 1/2 timestep behind fields, 
    # copy the up to date current into S
    @timeit to "source" @tturbo S .= S1
    # leapfrog plasma from n-1/2 to n+1/2 using fields at n
    @timeit to "advance!" advance!(plasma, grid1D, dt)
    # deposit current to grid
    @timeit to "deposit" depositcurrent!(grid1D, plasma)
    # now get the current at n+1/2
    @timeit to "source" sources!(S1, grid1D)
    # average to get current at n
    @timeit to "source" @tturbo S .= (S .+ S1) / 2
    # step fields from n to n+1
##    @timeit to "stepfields" stepfieldsHeun!(u, A, k1, k2, work, dt, S)
##    @timeit to "stepfields" stepfieldsEuler!(u, A, k1, work, dt, S)
    @timeit to "stepfields" stepfieldsRK4!(u, A, k1, k2, k3, k4, work, dt, S)
##    @timeit to "stepfields" bicgstabl!(u, CN⁻, mul!(work, CN⁺, u), reltol=1000eps())
    # write the field dofs back to the grid for next advance   
    @timeit to "dofs!" dofs!(grid1D, u)

     if i % ngifevery == 0 # only do this if we need to make plots
       #current = quadgk(x->currentfield(grid1D, [x], 1), a[1], b[1])[1]
       #meanfieldcurrentdensity = current / prod(b - a)
       #meanparticlecurrentdensity = sum(DGMaxwellPIC.xvelocity(plasma.species[1])) *
       #  weight / prod(b - a)
       #currentisconsistent = isapprox(meanfieldcurrentdensity, meanparticlecurrentdensity, atol=1e-8)

       x = DGMaxwellPIC.position(plasma.species[1])
       v = DGMaxwellPIC.velocity(plasma.species[1])
       p1 = scatter(x[1, :], v[1, :]); title!("$i of $NI")
       x = collect(1/NX/2:1/NX:1-1/NX/2)
       Ex = electricfield(grid1D, 1)
       @show i, extrema(Ex)#, currentisconsistent
       plot!(x, Ex, ylims=[-1,1])
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
