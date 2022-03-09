
struct Cell{N, T<:BasisFunctionType}
  state::State{N, T}
  min # Tuple NTuple Vector, StaticVector, but all N-long
  max
end
Base.in(x, c::Cell{N}) where N = all(i -> c.min[i] <= x[i] < c.max[i], 1:N)
dofshape(c::Cell) = dofshape(c.state)
jacobian(c::Cell{N}) where N = prod((c.max .- c.min)) / 2^N
state(c::Cell) = c.state
dofs(c::Cell) = vcat((c.state.q[i][:] for i in 1:6)...)
currentdofs(c::Cell) = vcat((c.state.q[i][:] for i in 7:9)...)
chargedofs(c::Cell) = c.state.q[10][:]
localx(c::Cell, x) = (x .- c.min) ./ (c.max .- c.min) .* 2 .- 1
boundingbox(c::Cell) = (c.min, c.max)
function dofs!(c::Cell, x)
  l = prod(dofshape(c))
  for i in 1:6, j in LinearIndices(dofshape(c))
    c.state.q[i][j] = x[(i-1)*l + j]
  end
  return c
end
zero!(c::Cell) = zero!(c.state)

