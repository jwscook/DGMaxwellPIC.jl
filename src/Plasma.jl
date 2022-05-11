struct Plasma
  species::Vector
end
Base.iterate(p::Plasma) = iterate(p.species)
Base.iterate(p::Plasma, i) = iterate(p.species, i)

function Base.sort!(p::Plasma, g::Grid)
  for s in p
    sort!(s, g)
  end
  return p
end

function depositcurrent!(g::Grid, plasma::Plasma)
  currentfielddofs!(g, 0) # zero current
  sort!(plasma, g)
  for species in plasma
    x = position(species)
    v = velocity(species)
    w = weight(species)
    q = charge(species)
    j, _ = workarrays(species)
    @tturbo for i in eachindex(j)
      j[i] = 0
    end
    j .= q .* (w' .* v) # create an eager array
    @assert !isempty(cellids(species))
    currentfield!(g, cellids(species), x, j) # add all current values cell-by-cell
    #for i in 1:numberofparticles(species) # add current values to cells particle-by-particle
    #  currentfield!(g, (@view x[:, i]), (@view j[:, i]))
    #end
  end
end

function currentloadvector!(output, g::Grid{N, T}, cellindex) where {N,T}
  cell = g[cellindex]
  nc = ndofs(cell, 1) # number of dofs per component
  @assert length(output) == 6nc "$(length(output)) vs $(6nc)"
  nodes = ndimnodes(g, cellindex)
  lumm = lumassmatrix!(g, cell)
  @views output[1:3nc] .= currentdofs(cell) # ∇×B = μJ + μϵ ∂E/∂t # yes electric current
  #@views output[3nc+1:6nc] .+= 0 # ∂B/∂t = - ∇×E # no magnetic current
  ldiv!(lumm, output)
end


function currentloadvector!(output, g::Grid{N,T}) where {N,T}
  @threads for i in CartesianIndices(g.data)
    cellindices = indices(g, Tuple(i))
    currentloadvector!((@view output[cellindices]), g, i)
  end
  return output 
end

function currentsource(g::Grid)
  output = zeros(ndofs(g))
  return currentloadvector!(output, g)
#  @threads for i in CartesianIndices(g.data)
#    cell = g[i]
#    inds = offsetindex(g, i) .+ electricfieldindices(cell)
#    output[inds] .= currentdofs(cell)
#  end
#  return output
end





function advance!(plasma::Plasma, g::Grid{N}, dt) where {N}
  lb = lower(g)
  ub = upper(g)
  for species in plasma
    q_m = charge(species) / mass(species)
    x = position(species)
    v = velocity(species)
    E, B = workarrays(species)
    @tturbo for i in eachindex(E)
      E[i] = 0
      B[i] = 0
    end
    cids = cellids(species) 
    @inbounds @views @threads for i in 1:numberofparticles(species)
      advect!(x[:, i], v[:, i], dt/2)
      @. x[:, i] = mod(x[:, i] - lb, ub - lb) + lb
    end
    sort!(species, g)
    t1 = @spawn electricfield!(E, g, cids, x) # TODO should get E and B at the same time!
    t2 = @spawn magneticfield!(B, g, cids, x)
    wait(t1)
    wait(t2)
    @inbounds @views @threads for i in 1:numberofparticles(species)
      E[:, i] .*= q_m
      B[:, i] .*= q_m
      borispush!(x[:, i], v[:, i], SVector{3,Float64}(E[:, i]), SVector{3,Float64}(B[:, i]), dt, Val(N))
      advect!(x[:, i], v[:, i], dt/2)
      @. x[:, i] = mod(x[:, i] - lb, ub - lb) + lb
    end
    sort!(species, g)
  end
  return plasma
end


