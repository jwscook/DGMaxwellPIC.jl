
_fluxmatrix(stencil, x) = sparse(kron(stencil, x))
upwindfluxmatrix(::Val{1}, x::AbstractArray) = _fluxmatrix((@SArray [0 0 0; -1 1 0; -1 0 1]), x)
upwindfluxmatrix(::Val{2}, x::AbstractArray) = _fluxmatrix((@SArray [1 -1 0; 0 0 0; 0 -1 1]), x)
upwindfluxmatrix(::Val{3}, x::AbstractArray) = _fluxmatrix((@SArray [1 0 -1; 0 1 -1; 0 0 0]), x)
@memoize upwindfluxmatrix(::Val{T}, n::Integer) where {T} = fluxmatrix(Val(T), I(n))
upwindfluxmatrix(s::State{N}, a) where N = sum(upwindfluxmatrix(Val(i), a) for i in 1:N)

fluxmatrix(::Val{1}, x::AbstractArray) = _fluxmatrix((@SArray [0 0 0; 0 0 1; 0 -1 0]), x)
fluxmatrix(::Val{2}, x::AbstractArray) = _fluxmatrix((@SArray [0 0 1; 0 0 0; -1 0 0]), x)
fluxmatrix(::Val{3}, x::AbstractArray) = _fluxmatrix((@SArray [0 -1 0; 1 0 0; 0 0 0]), x)
@memoize fluxmatrix(::Val{T}, n::Integer) where {T} = fluxmatrix(Val(T), I(n))

function surfacefluxmassmatrix(cell::Cell, nodes::NDimNodes, dim, side::FaceDirection, upwind=0.0)
  output = spzeros(ndofs(cell), ndofs(cell))
  return surfacefluxmassmatrix!(output, cell, nodes, dim, side, upwind)
end

function surfacefluxmassmatrix!(output, cell::Cell, nodes::NDimNodes, dim, side::FaceDirection,
    upwind=0.0)
  @assert size(output) == (ndofs(cell), ndofs(cell))
  nc = ndofs(cell, 1) # number of dofs per component
  inds = facedofindices(nodes, dim, side)
  sfmm = surfacefluxmassmatrix(nodes, nodes, dim, side) * jacobian(cell)
  @assert size(sfmm) == (nc, nc)
  fm = fluxmatrix(Val(dim), sfmm)
  @views output[1:3nc, 3nc+1:6nc] .+= fm .* speedoflight^2
  @views output[3nc+1:6nc, 1:3nc] .-= fm
  if !iszero(upwind)
    inds = facedofindices(nodes, dim, side)
    fm = upwindfluxmatrix(Val(dim), sfmm) .* speedoflight * upwind / 2
    @views output[1:3nc, 1:3nc] .+= fm .* epsilon0
    @views output[3nc+1:6nc, 3nc+1:6nc] .-= fm
  end
  return output
end

function volumefluxmassmatrix(cell::Cell{N}, nodes::NDimNodes) where {N}
  ns = ndofs(cell)
  output = spzeros(ns, ns)
  nc = ndofs(cell, 1) # number of dofs per component
  for dim in 1:N
    fmm = volumefluxmassmatrix(nodes, nodes, dim) * jacobian(cell)
    fm = fluxmatrix(Val(dim), fmm)
    @views output[1:3nc, 3nc+1:6nc] .+= fm .* speedoflight^2
    @views output[3nc+1:6nc, 1:3nc] .-= fm
  end
  return output
end

function volumemassmatrix(_::Union{Cell, State}, nodes::NDimNodes)
  return kron(I(6), massmatrix(nodes))
end

function assembler(g::Grid{N,T}, f::F) where {N,T, F}
  n = ndofs(g)
  output = spzeros(n,n)
  for i in CartesianIndices(g.data)
    cellindices = indices(g, Tuple(i))
    nodes = NDimNodes(dofshape(g[i]), T)
    @views output[cellindices, cellindices] .= f(g[i], nodes)
  end
  return output 
end
function volumefluxmassmatrix(g::Grid{N,T}) where {N,T}
  return assembler(g, volumefluxmassmatrix)
end

function volumemassmatrix(g::Grid{N,T}) where {N,T}
  return assembler(g, volumemassmatrix)
end

@memoize function indices(g::Grid{N,T}, cellindices) where {N,T,F}
  return (1:ndofs(g[cellindices...])) .+ offsetindex(g, cellindices)
end

function findneighbour(g::Grid{N}, homeindex, searchdir, side) where N
  gridsize = size(g)
  return [mod1(homeindex[j] + (searchdir == j) * ((side == Low) ? -1 : 1), gridsize[j]) for j in 1:N]
end

function surfacefluxmassmatrix(g::Grid{N,T}, upwind=0.0) where {N,T}
  gridsize = size(g)
  n = ndofs(g)
  output = spzeros(n,n)
  for cartindex in CartesianIndices(g.data)
    cellindex = Tuple(cartindex)
    cell = g[cartindex]
    nodes = NDimNodes(dofshape(cell), T)
    celldofindices = indices(g, cellindex)
    for dim in 1:N, (side, factor) in ((Low, -1), (High, 1))
      neighbourdofindices = indices(g, findneighbour(g, cellindex, dim, side))
      flux = factor .* surfacefluxmassmatrix(cell, nodes, dim, side, upwind) * jacobian(cell)
      @views output[neighbourdofindices, celldofindices] .+= flux
      @views output[celldofindices, celldofindices] .-= flux
    end
  end
  return output 
end

function assemble(g::Grid{N,T}; upwind=0.0) where {N, T}
  return volumemassmatrix(g) \ (volumefluxmassmatrix(g) .+ surfacefluxmassmatrix(g, upwind))
end



