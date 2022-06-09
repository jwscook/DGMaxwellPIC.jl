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
dofsview(s::State{N}, c) where {N} = selectdim(s.q, N+1, c)
# doing @view by hand is faster
dofsview(s::State{1}, c) = @view s.q[:, c]
dofsview(s::State{2}, c) = @view s.q[:, :, c]
dofsview(s::State{3}, c) = @view s.q[:, :, :, c]

workdofs(s::State) = dofsview(s, 11)

dofs(s::State, components::Union{Int,UnitRange{Int}}=1:6) = dofsview(s, components)

function dofs!(s::State, x, components=1:6)
  dofsop!(s, x, components, setop!)
  return s
end

function incrementdofs!(s::State, x, components=1:6)
  dofsop!(s, x, components, incrementop!)
  return s
end

@inline function setop!(lhs, indexlhs, rhs::Number, _=missing)
  @inbounds lhs[indexlhs] = rhs
end
@inline setop!(lhs, indexlhs, rhs, indexrhs) = setop!(lhs, indexlhs, rhs[indexrhs])

@inline function incrementop!(lhs, indexlhs, rhs::Number, _=missing)
  @inbounds lhs[indexlhs] += rhs
end
@inline incrementop!(lhs, indexlhs, rhs, indexrhs) = incrementop!(lhs, indexlhs, rhs[indexrhs])

function dofsop!(s::State, x, components, op!::F) where F
  l = prod(dofshape(s))
  for (i, c) in enumerate(components)
    ii = prod(dofshape(s)) * (c - 1)
    li_l = (i - 1) * l
    @inbounds for j in 1:prod(dofshape(s))
      op!(s.q, j + ii, x, li_l + j)
    end
  end
  return s
end


