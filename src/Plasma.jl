struct Plasma{T}
  species::Vector{T}
end
Base.iterate(p::Plasma) = iterate(p.species)
Base.iterate(p::Plasma, i) = iterate(p.species, i)

function Base.sort!(p::Plasma, g::Grid)
  for s in p
    sort!(s, g)
  end
  return p
end

function depositcurrent!(g::Grid{N}, plasma::Plasma) where N
  currentfielddofs!(g, 0) # zero current
  sort!(plasma, g)
  for species in plasma
    v = velocity(species)
    w = weight(species)
    q = charge(species)
    j, _ = workarrays(species)
    @threads for jj in 1:size(v, 2)
       @inbounds for i in 1:N
        j[i, jj] = q * w[jj] * v[i, jj]
      end
    end
    cids = cellids(species)
    x = position(species)
    currentfield!(g, cids, x, j) # add all current values cell-by-cell
  end
end

function currentloadvector!(output, g::Grid{N, T}, cellindex) where {N,T}
  cell = g[cellindex]
  nc = ndofs(cell, 1) # number of dofs per component
  @assert length(output) == 6nc "$(length(output)) vs $(6nc)"
  nodes = ndimnodes(g, cellindex)
  lumm = lumassmatrix!(g, cell)
  @views output[1:3nc] .= currentdofs(cell) # ∇×B = μJ + μϵ ∂E/∂t # yes electric current
  #No-op #@views output[3nc+1:6nc] .= 0 # ∂B/∂t = - ∇×E # no magnetic current
  ldiv!(lumm, output)
end


function currentloadvector!(output, g::Grid{N,T}) where {N,T}
  @threads for i in CartesianIndices(g.data)
    cellindices = indices(g, Tuple(i))
    currentloadvector!((@view output[cellindices]), g, i)
  end
  return output 
end

function currentsource!(output, g::Grid)
  return currentloadvector!(output, g)
end


function advance!(plasma::Plasma, g::Grid{N}, dt) where {N}
  lb = lower(g)
  ub = upper(g)
  for species in plasma
    q_m = charge(species) / mass(species)
    x = position(species)
    v = velocity(species)
    _, EB = workarrays(species)
    @inbounds @views @threads for i in 1:numberofparticles(species)
      EB[:, i] .= zero(eltype(EB))
      advect!(x[:, i], v[:, i], dt/2)
      @. x[:, i] = mod(x[:, i] - lb, ub - lb) + lb
    end
    sort!(species, g) # sort into cells to get EB field efficiently
    electromagneticfield!(EB, g, cellids(species), x)
    @inbounds @views @threads for i in 1:numberofparticles(species)
      EB[:, i] .*= q_m
      qE_m = SVector{3, Float64}(EB[1:3, i])
      qB_m = SVector{3, Float64}(EB[4:6, i])
      borispush!(x[:, i], v[:, i], qE_m, qB_m, dt, Val(N))
      advect!(x[:, i], v[:, i], dt/2)
      @. x[:, i] = mod(x[:, i] - lb, ub - lb) + lb
    end
    sort!(species, g) # sort again after move
  end
  return plasma
end


