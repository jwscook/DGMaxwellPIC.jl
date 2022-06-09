# Each N-dimension array represents the values at the nodes as they are arranged
# in N-dimensionsal space
struct State{N, T<:BasisFunctionType, N⁺¹}
  q::Array{Float64, N⁺¹}
  dofshape::NTuple{N, Int}
  nodes::NDimNodes
  function State(ncoeffsperdim, ::Type{T}) where {T<:BasisFunctionType}
    dofs = zeros(ncoeffsperdim..., 11)
    nodes = NDimNodes(ncoeffsperdim, T)
    N = length(ncoeffsperdim)
    N⁺¹ = N + 1
    return new{N, T, N⁺¹}(dofs, Tuple(ncoeffsperdim), nodes) # need only be 6?
  end
end
Base.size(s::State) = s.dofshape
dofshape(s::State) = s.dofshape
state(s::State) = s
nodes(s::State) = s.nodes
workdofs(s::State{N}) where {N} = selectdim(s.q, N+1, 11)

function dofs(s::State{N}, components::Union{Int,UnitRange{Int}}=1:6) where N
  return selectdim(s.q, N+1, components)
end

function dofs!(s::State{N}, x, component::Integer) where N
  selectdim(s.q, N+1, component) .= x
end

function dofs!(s::State, x::AbstractArray, components::UnitRange{Int}=1:6)
  dofsop!(s, x, components, setop!)
  return s
end

function incrementdofs!(s::State, x, components::UnitRange{Int}=1:6)
  dofsop!(s, x, components, incrementop!)
  return s
end

function setop!(lhs, indexlhs, rhs, indexrhs)
  lhs[indexlhs] = rhs[indexrhs]
end

function incrementop!(lhs, indexlhs, rhs, indexrhs)
  lhs[indexlhs] += rhs[indexrhs]
end

function dofsop!(s::State{N}, x::AbstractArray, components::UnitRange{Int}, op!::F) where {N, F}
  l = prod(dofshape(s))
  for (i, c) in enumerate(components)
    #sqi = selectdim(s.q, N+1, c)
    ii = prod(dofshape(s)) * (c - 1) + 1
    li_l = (i - 1) * l
    @inbounds for j in LinearIndices(dofshape(s))
      #op!(sqi, j, x, li_l + j)
      op!(s.q, j + ii, x, li_l + j)
    end
  end
  return s
end

function incrementdofs!(s::State{N}, x, component::Integer) where N
  selectdim(s.q, N+1, component) .+= x
end


