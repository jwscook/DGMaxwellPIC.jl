
struct Cell{N, T<:BasisFunctionType, U}
  state::State{N, T}
  lower::U # Tuple NTuple Vector, StaticVector, but all N-long
  upper::U
  inverselengths::U
  function Cell(state::State{N,T}, lower::U, upper::U) where {N,T,U}
    return new{N,T,U}(state, lower, upper, 1.0 ./ (upper .- lower))
  end
end
Base.in(x::Number, c::Cell{1}) = @inbounds c.lower[1] <= x < c.upper[1]
Base.in(x, c::Cell{1}) = @inbounds c.lower[1] <= x[1] < c.upper[1]
Base.in(x, c::Cell{2}) = @inbounds (c.lower[1] <= x[1] < c.upper[1]) &&
                                   (c.lower[2] <= x[2] < c.upper[2])
Base.in(x, c::Cell{3}) = @inbounds (c.lower[1] <= x[1] < c.upper[1]) &&
                                   (c.lower[2] <= x[2] < c.upper[2]) &&
                                   (c.lower[3] <= x[3] < c.upper[3])
_in(x, c::Cell, i) = @inbounds c.lower[i] <= x[i] < c.upper[i]
Base.in(x, c::Cell{N}) where N = all(i -> _in(x, c, i), 1:N)
dofshape(c::Cell) = dofshape(c.state)
function jacobian(c::Cell{N}; ignore=0) where N
  output = 1.0
  for i in 1:N
    i == ignore && continue
    output *= (c.upper[i] - c.lower[i]) / 2 # factor of 1/2 to convert from reference cell [-1,1]
  end
  return output
end

state(c::Cell) = c.state
dofs(c::Cell) = vcat(((@view c.state.q[i][:]) for i in 1:6)...) # TODO think about shape
currentdofs(c::Cell) = vcat(((@view c.state.q[i][:]) for i in 7:9)...) # TODO think about shape
chargedofs(c::Cell) = @view c.state.q[10]
workdofs(c::Cell) = @view c.state.q[11]
referencex(x, c::Cell{N}) where {N} = @. ((x - c.lower) * c.inverselengths * 2 - 1)
originalx(x, c::Cell{N}) where {N} = @. ((x + 1) /2 * (c.upper - c.lower) + c.lower)
referencex!(x, c::Cell{N}) where {N} = (@. x = (x - c.lower) * c.inverselengths * 2 - 1)
originalx!(x, c::Cell{N}) where {N} = (@. x = (x + 1) /2 * (c.upper - c.lower) + c.lower)
boundingbox(c::Cell) = (c.lower, c.upper)
lower(c::Cell) = c.lower
upper(c::Cell) = c.upper
centre(c::Cell) = (lower(c) .+ upper(c)) ./ 2
dofs!(c::Cell, x) = dofs!(state(c), x)
zero!(c::Cell) = zero!(c.state)

massmatrix(c::Cell{N,T}) where {N,T} = massmatrix(NDimNodes(dofshape(c), T)) * jacobian(c)
