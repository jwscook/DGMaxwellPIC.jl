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
zero!(s::State) = fill!(s.q, 0)
state(s::State) = s
nodes(s::State) = s.nodes
workdofs(s::State{N}) where {N} = selectdim(s.q, N, dofsindices(s, 11))

function dofs(s::State{N}, components::Union{Int,UnitRange{Int}}=1:6) where N
  inds = minimum(dofsindices(s, minimum(components))):maximum(dofsindices(s, maximum(components)))
  return selectdim(s.q, N, inds)
end

function dofs!(s::State{N}, x::Number, components=1:16) where N
  for i in components
    selectdim(s.q, N, dofsindices(s, i)) .= x
  end
  return s
end

function dofs!(s::State{N}, x::AbstractArray, components=1:6) where N
  for i in components
    sqi = selectdim(s.q, N, dofsindices(s, i))
    for j in LinearIndices(dofshape(s))
      sqi[j] = x[j]
    end
  end
  return s
end

function incremementdofs!(s::State{N}, x, component::Integer) where N
  sqi = selectdim(s.q, N, dofsindices(s, component))
  qi[j] .+= x
  return s
end

