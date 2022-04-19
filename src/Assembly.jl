
@memoize function indices(g::Grid{N,T}, cellindices) where {N,T,F}
  # TODO fix need to index by cellindices... i.e. with splat
  return (1:ndofs(g[cellindices...])) .+ offsetindex(g, cellindices)
end

function findneighbourgridindex(g::Grid{N}, homeindex, searchdir, side) where N
  gridsize = size(g)
  return [mod1(homeindex[j] + (searchdir == j) * ((side == Low) ? -1 : 1), gridsize[j]) for j in 1:N]
end

_fluxmatrix(stencil, x::AbstractArray) = sparse(kron(stencil, x))
upwindfluxmatrix(::Val{1}, x) = _fluxmatrix((@SArray [0 0 0; 0 1 0; 0 0 1]), x)
upwindfluxmatrix(::Val{2}, x) = _fluxmatrix((@SArray [1 0 0; 0 0 0; 0 0 1]), x)
upwindfluxmatrix(::Val{3}, x) = _fluxmatrix((@SArray [1 0 0; 0 1 0; 0 0 0]), x)
#@memoize upwindfluxmatrix(::Val{T}, n::Integer) where {T} = fluxmatrix(Val(T), I(n))
#upwindfluxmatrix(s::State{N}, a) where N = sum(upwindfluxmatrix(Val(i), a) for i in 1:N)

fluxmatrix(::Val{1}, x) = _fluxmatrix((@SArray [0 0 0; 0 0 1; 0 -1 0]), x)
fluxmatrix(::Val{2}, x) = _fluxmatrix((@SArray [0 0 -1; 0 0 0; 1 0 0]), x)
fluxmatrix(::Val{3}, x) = _fluxmatrix((@SArray [0 1 0; -1 0 0; 0 0 0]), x)
#@memoize fluxmatrix(::Val{T}, n::Integer) where {T} = fluxmatrix(Val(T), I(n))

function surfacefluxstiffnessmatrix(cell::Cell, nodes::NDimNodes, dim, side::FaceDirection, upwind=0.0)
  output = spzeros(ndofs(cell), ndofs(cell))
  return surfacefluxstiffnessmatrix!(output, cell, nodes, dim, side, upwind)
end

function surfacefluxstiffnessmatrix!(output, cell::Cell, nodes::NDimNodes, dim, side::FaceDirection,
    upwind=0.0)
  @assert size(output) == (ndofs(cell), ndofs(cell))
  nc = ndofs(cell, 1) # number of dofs per component
  sfmm = surfacefluxstiffnessmatrix(nodes, nodes, dim, side)
  @assert size(sfmm) == (nc, nc) "$(size(sfmm)) != ($nc, $nc)"
  fm = fluxmatrix(Val(dim), sfmm)
  @views output[1:3nc, 3nc+1:6nc] .+= fm .* speedoflight^2
  @views output[3nc+1:6nc, 1:3nc] .-= fm
  if !iszero(upwind)
    ufm = upwindfluxmatrix(Val(dim), sfmm) * upwind
    @views output[1:3nc, 1:3nc] .+= ufm .* epsilon0
    @views output[3nc+1:6nc, 3nc+1:6nc] .+= ufm
  end
  return output
end

function surfacefluxstiffnessmatrix(g::Grid{N,T}, upwind=0.0) where {N,T}
  gridsize = size(g)
  n = ndofs(g)
  output = spzeros(n,n)
  for cartindex in CartesianIndices(g.data)
    cellindex = Tuple(cartindex)
    cell = g[cartindex]
    nodes = NDimNodes(dofshape(cell), T)
    celldofindices = indices(g, cellindex)
    for dim in 1:N, (side, factor) in ((Low, -1), (High, 1))
      flux = surfacefluxstiffnessmatrix(cell, nodes, dim, side, upwind)
      flux .*= factor * jacobian(cell; ignore=dim)
      @views output[celldofindices, celldofindices] .-= flux

      neighbourcellgridindex = findneighbourgridindex(g, cellindex, dim, opposite(side))
      neighbourcell = g[neighbourcellgridindex...]
      flux = surfacefluxstiffnessmatrix(neighbourcell, nodes, dim, opposite(side), upwind)
      flux .*= factor * jacobian(cell; ignore=dim)
      neighbourdofindices = indices(g, neighbourcellgridindex)
      @views output[celldofindices, neighbourdofindices] .+= flux
    end
  end
  return output
end


function volumefluxstiffnessmatrix(cell::Cell{N}, nodes::NDimNodes) where {N}
  ns = ndofs(cell)
  output = spzeros(ns, ns)
  nc = ndofs(cell, 1) # number of dofs per component
  for dim in 1:N
    fmm = volumefluxstiffnessmatrix(nodes, nodes, dim) * jacobian(cell)
    fm = fluxmatrix(Val(dim), fmm)
    @views output[1:3nc, 3nc+1:6nc] .+= fm .* speedoflight^2
    @views output[3nc+1:6nc, 1:3nc] .-= fm
  end
  return output
end

function volumemassmatrix(cell::Cell, nodes::NDimNodes)
  return kron(I(6), massmatrix(nodes)) * jacobian(cell)
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
function volumefluxstiffnessmatrix(g::Grid{N,T}) where {N,T}
  return assembler(g, volumefluxstiffnessmatrix)
end

function volumemassmatrix(g::Grid{N,T}) where {N,T}
  return assembler(g, volumemassmatrix)
end

function assemble(g::Grid{N,T}; upwind=0.0) where {N, T}
  return Matrix(volumemassmatrix(g)) \ (volumefluxstiffnessmatrix(g) .+ surfacefluxstiffnessmatrix(g, upwind))
end



