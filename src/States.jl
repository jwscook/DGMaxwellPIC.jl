# Each N-dimension array represents the values at the nodes as they are arranged
# in N-dimensionsal space
struct State{N, T<:BasisFunctionType}
  q::Array{Float64, N}
  dofshape::NTuple{N, Int}
  nodes::NDimNodes
  function State(ncoeffsperdim, ::Type{T}) where {T<:BasisFunctionType}
    N = length(ncoeffsperdim)
    repeats = [i == N ? 11 : 1 for i in 1:N]
    dofs = repeat(zeros(ncoeffsperdim...), repeats...)
    nodes = NDimNodes(ncoeffsperdim, T)
    return new{N, T}(dofs, Tuple(ncoeffsperdim), nodes) # need only be 6?
  end
end
dofsindices(s::State, i::Int) = (s.dofshape[end] .* (i - 1) + 1):(s.dofshape[end] .* i)
Base.size(s::State) = s.dofshape
dofshape(s::State) = s.dofshape
state(s::State) = s
nodes(s::State) = s.nodes
workdofs(s::State{N}) where {N} = selectdim(s.q, N, dofsindices(s, 11))

function dofs(s::State{N}, components::Union{Int,UnitRange{Int}}=1:6) where N
  inds = minimum(dofsindices(s, minimum(components))):maximum(dofsindices(s, maximum(components)))
  return selectdim(s.q, N, inds)
end

function dofs!(s::State{N}, x, component::Integer) where N
  selectdim(s.q, N, dofsindices(s, component)) .= x
end

function dofs!(s::State{N}, x::AbstractArray, components::UnitRange{Int}=1:6) where N
  l = prod(dofshape(s))
  for (i, c) in enumerate(components)
    sqi = selectdim(s.q, N, dofsindices(s, c))
    for j in LinearIndices(dofshape(s))
      sqi[j] = x[(i-1)*l+j]
    end
  end
  return s
end

function incremementdofs!(s::State{N}, x::AbstractArray, components::UnitRange{Int}=1:6) where N
  l = prod(dofshape(s))
  for (i, c) in enumerate(components)
    sqi = selectdim(s.q, N, dofsindices(s, c))
    for j in LinearIndices(dofshape(s))
      sqi[j] += x[(i-1)*l+j]
    end
  end
  return s
end

function incrementdofs!(s::State, x, components::UnitRange{Int}=1:6)
  l = prod(dofshape(s))
  for i in components
    sqi = s.q[i]
    for j in LinearIndices(dofshape(s))
      sqi[j] += x[(i-1)*l + j]
    end
  end
  return s
end

