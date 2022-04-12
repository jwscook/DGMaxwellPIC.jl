
struct Cell{N, T<:BasisFunctionType, U}
  state::State{N, T}
  lower::U # Tuple NTuple Vector, StaticVector, but all N-long
  upper::U
end
Base.in(x::Number, c::Cell{1}) = c.lower[1] <= x < c.upper[1]
Base.in(x, c::Cell{N}) where N = all(i -> c.lower[i] <= x[i] < c.upper[i], 1:N)
dofshape(c::Cell) = dofshape(c.state)
function jacobian(c::Cell{N}; ignore=0) where N
  output = 1.0
  for i in 1:N
    i == ignore && continue
    output *= (c.upper[i] - c.lower[i]) / 2
  end
  return output
end

state(c::Cell) = c.state
dofs(c::Cell) = vcat(((@view c.state.q[i][:]) for i in 1:6)...) # TODO think about shape
currentdofs(c::Cell) = vcat(((@view c.state.q[i][:]) for i in 7:9)...) # TODO think about shape
chargedofs(c::Cell) = @view c.state.q[10]
workdofs(c::Cell) = @view c.state.q[11]
referencex(c::Cell, x) = (x .- c.lower) ./ (c.upper .- c.lower) .* 2 .- 1
originalx(c::Cell, x) = (x .+ 1) ./2 .* (c.upper .- c.lower) .- c.lower
boundingbox(c::Cell) = (c.lower, c.upper)
lower(c::Cell) = c.lower
upper(c::Cell) = c.upper
centre(c::Cell) = (lower(c) .+ upper(c)) ./ 2
dofs!(c::Cell, x) = dofs!(state(c), x)
zero!(c::Cell) = zero!(c.state)


