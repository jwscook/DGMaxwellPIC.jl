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

@memoize function offsetindex(g::Grid{N,T}, cellindices) where {N,T,F}
  tcellindices = Tuple(cellindices)
  @assert all(ones(N) .<= tcellindices .<= size(g.data))
  offset = 0
  for i in CartesianIndices(g.data)
    if all(j->isequal(j...), zip(tcellindices, Tuple(i))) # i == cellindices
      return offset
    end
    offset += ndofs(g[i]) # not the cell we want so count up all the number of dofs in the offset
  end
  throw(ErrorException("Shouldn't be able to get here"))
end


# should these belong here?
for (fname, rnge) in ((:ndofs, 1:6), (:currentndofs, 7:9), (:totalndofs, 1:10))
  @eval $(fname)(u::State) = sum(prod(size(u.q[i])) for i in $(rnge))
  @eval $(fname)(u::State, components) = sum(prod(size(u.q[component])) for component in components)
  @eval $(fname)(c::Cell) = $(fname)(c.state)
  @eval $(fname)(c::Cell, i) = $(fname)(c.state, i)
  @eval $(fname)(g::Grid) = sum($(fname)(i) for i in g)
end

# Should these belong here?
# This isn't very efficient
electricfieldindices(s::Union{State,Cell}) = 1:ndofs(s, 1:3)
magneticfieldindices(s::Union{State,Cell}) = ndofs(s, 1:3) .+ (1:ndofs(s, 4:6))
currentfieldindices(s::Union{State,Cell}) = ndofs(s, 1:6) .+ (1:ndofs(s, 7:9))
chargefieldindices(s::Union{State,Cell}) = ndofs(s, 1:9) .+ (1:ndofs(s, 10))

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

function currentsource(grid::Grid)
  output = zeros(ndofs(grid))
  for i in CartesianIndices(grid.data)
    cell = grid[i]
    inds = offsetindex(grid, i) .+ electricfieldindices(cell)
    output[inds] .= currentdofs(cell)
  end
  return output
end

