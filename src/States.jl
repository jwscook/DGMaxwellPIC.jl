# Each N-dimension array represents the values at the nodes as they are arranged
# in N-dimensionsal space
struct State{N, T<:BasisFunctionType}
  q::Vector{Array{Float64, N}}
  dofshape::NTuple{N,Int}
  nodes::NDimNodes
  function State(ncoeffsperdim, ::Type{T}) where {T<:BasisFunctionType}
    N = length(ncoeffsperdim)
    vecofdofs = [zeros(ncoeffsperdim...) for i in 1:11]
    nodes = NDimNodes(ncoeffsperdim, T)
    return new{N, T}(vecofdofs, Tuple(ncoeffsperdim), nodes) # need only be 6?
  end
end
Base.size(s::State) = s.dofshape
dofshape(s::State) = s.dofshape
numcomponents(s::State) = 11
zero!(s::State) = fill!.(s.q, 0)
state(s::State) = s
nodes(s::State) = s.nodes
workdofs(s::State) = s.q[11]

function dofs(s::State, component::Integer)
  return s.q[component]
end


function dofs!(s::State, x::Number)
  for i in 1:6
    s.q[i] .= x
  end
  return s
end

function dofs!(s::State, x, component::Integer)
  sqi = s.q[component]
  for j in LinearIndices(dofshape(s))
    sqi[j] = x[j]
  end
  return s
end

function dofs!(s::State, x, components::UnitRange{Int}=1:6)
  l = prod(dofshape(s))
  for i in components
    sqi = s.q[i]
    for j in LinearIndices(dofshape(s))
      sqi[j] = x[(i-1)*l + j]
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

