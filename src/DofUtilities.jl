varsize(state::State, component) = size(state.q[component])
varsize(cell::Cell, component) = varsize(cell.state, component)

index(sizetuple, dim::Integer, side::FaceDirection) = side == Low ? 1 : sizetuple[dim]
# return the indices of the nodes that belong to the face on the side (Low or High) in dimension dim
function facedofindices(cs::V, component, dim::Int, side::FaceDirection) where {N,T,V<:Union{Cell{N,T},State{N,T}}}
  offset = componentindexoffset(cs, component)
  return facedofindices(varsize(cs, component), dim, side) .+ offset
end
facedofindices(s::V, dim::Int, side::FaceDirection) where {N,T,V<:Union{Cell{N,T},State{N,T}}} = facedofindices(s, 1, dim, side)

function facedofindices(sizetuple::Tuple, dim::Int, side::FaceDirection)
  1 <= dim <= length(sizetuple) || throw(ArgumentError("Dim not a physical dimension, $dim"))
  linear = LinearIndices(sizetuple)
  ind = index(sizetuple, dim, side)
  indices = Vector{Union{Colon, Int}}([i == dim ? ind : Colon() for i in 1:length(sizetuple)])
  return linear[indices...]
end

for (fname, rnge) in ((:ndofs, 1:6), (:currentndofs, 7:9))
  @eval $(fname)(u::State) = sum(prod(size(u.q[i])) for i in $(rnge))
  @eval $(fname)(u::State, component) = prod(size(u.q[component]))
  @eval $(fname)(c::Cell) = $(fname)(c.state)
  @eval $(fname)(c::Cell, i) = $(fname)(c.state, i)
  @eval $(fname)(g::Grid) = sum($(fname)(i) for i in g)
end

function dofs(grid::Grid)
  output = zeros(ndofs(grid))
  for i in CartesianIndices(grid.data)
    cell = grid[i]
    output[indices(grid, Tuple(i))] .= dofs(cell)
  end
  return output
end

function dofs!(grid::Grid, x)
  output = zeros(ndofs(grid))
  for i in CartesianIndices(grid.data)
    cell = grid[i]
    dofs!(cell, x[indices(grid, Tuple(i))])
  end
  return grid
end

function currentdofs(grid::Grid)
  output = zeros(currentndofs(grid))
  for i in CartesianIndices(grid.data)
    cell = grid[i]
    output[indices(grid, Tuple(i), currentndofs)] .= currentdofs(cell)
  end
  return output
end


