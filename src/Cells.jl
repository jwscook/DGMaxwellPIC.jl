
struct Cell{N, T<:BasisFunctionType}
  state::State{N, T}
  lower # Tuple NTuple Vector, StaticVector, but all N-long
  upper
end
Base.in(x::Number, c::Cell{1}) = c.lower[1] <= x < c.upper[1]
Base.in(x, c::Cell{N}) where N = all(i -> c.lower[i] <= x[i] < c.upper[i], 1:N)
dofshape(c::Cell) = dofshape(c.state)
jacobian(c::Cell{N}) where N = prod((c.upper .- c.lower)) / 2^N
state(c::Cell) = c.state
dofs(c::Cell) = vcat((c.state.q[i][:] for i in 1:6)...)
currentdofs(c::Cell) = vcat((c.state.q[i][:] for i in 7:9)...)
chargedofs(c::Cell) = c.state.q[10][:]
localx(c::Cell, x) = (x .- c.lower) ./ (c.upper .- c.lower) .* 2 .- 1
boundingbox(c::Cell) = (c.lower, c.upper)
lower(c::Cell) = c.lower
upper(c::Cell) = c.upper
centre(c::Cell) = (lower(c) .+ upper(c)) ./ 2
function dofs!(c::Cell, x)
  l = prod(dofshape(c))
  for i in 1:6, j in LinearIndices(dofshape(c))
    c.state.q[i][j] = x[(i-1)*l + j]
  end
  return c
end
dofs!(c::Cell, x) = dofs!(state(c), x)
zero!(c::Cell) = zero!(c.state)

