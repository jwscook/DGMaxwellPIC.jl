varsize(state::State, component) = size(state.q[component])
varsize(cell::Cell, component) = varsize(cell.state, component)

index(sizetuple, dim::Integer, side::FaceDirection) = side == Low ? 1 : sizetuple[dim]

function facedofindices(nodes::NDimNodes, dim::Int, side::FaceDirection)
  sizetuple = size(nodes)
  1 <= dim <= length(sizetuple) || throw(ArgumentError("Dim not a physical dimension, $dim"))
  linear = LinearIndices(sizetuple)
  ind = index(sizetuple, dim, side)
  if nodes[dim] isa LobattoNodes
    indices = Vector{Union{Colon, Int}}([i == dim ? ind : Colon() for i in 1:length(sizetuple)])
    return linear[indices...]
  else
    return linear[:]
  end
end

@memoize function offsetindex(g::Grid{N,T}, cellindices) where {N,T,F}
  tcellindices = Tuple(cellindices)
  @assert all(1 <= tcellindices[i] <= size(g.data, i) for i in eachindex(tcellindices))
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

function dofs!(grid::Grid, x::Number)
  for cell in grid
    dofs!(cell, x)
  end
  return grid
end

function dofs!(grid::Grid, x::AbstractArray)
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

