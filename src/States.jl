# Each N-dimension array represents the values at the nodes as they are arranged
# in N-dimensionsal space
struct State{N, T<:BasisFunctionType}
  q::Vector{Array{Float64, N}}
  function State(ncoeffsperdim, ::Type{T}) where {T<:BasisFunctionType}
    N = length(ncoeffsperdim)
    return new{N, T}([zeros(ncoeffsperdim...) for i in 1:10]) # need only be 6?
  end
end
Base.size(s::State) = size(s.q[1])
dofshape(s::State) = size(s.q[1])
zero!(s::State) = fill!.(s.q, 0)

function dofs!(s::State, x)
  l = prod(dofshape(s))
  for i in 1:6, j in LinearIndices(dofshape(s))
    s.q[i][j] = x[(i-1)*l + j]
  end
  return s
end

