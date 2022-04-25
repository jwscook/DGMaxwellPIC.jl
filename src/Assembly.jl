
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

# positive curl / levi-civita
fluxmatrix(::Val{1}, x) = _fluxmatrix((@SArray [0 0 0; 0 0 1; 0 -1 0]), x)
fluxmatrix(::Val{2}, x) = _fluxmatrix((@SArray [0 0 -1; 0 0 0; 1 0 0]), x)
fluxmatrix(::Val{3}, x) = _fluxmatrix((@SArray [0 1 0; -1 0 0; 0 0 0]), x)
#@memoize fluxmatrix(::Val{T}, n::Integer) where {T} = fluxmatrix(Val(T), I(n))

function surfacefluxstiffnessmatrix(cell::Cell, nodes::NDimNodes, dim, side::FaceDirection, rev::Bool, upwind=0.0)
  output = zeros(ndofs(cell), ndofs(cell))
  return surfacefluxstiffnessmatrix!(output, cell, nodes, dim, side, rev, upwind)
end

function surfacefluxstiffnessmatrix!(output, cell::Cell, nodes::NDimNodes, dim, side::FaceDirection, rev::Bool,
    upwind=0.0)
  @assert size(output) == (ndofs(cell), ndofs(cell))
  nc = ndofs(cell, 1) # number of dofs per component
  sfmm = surfacefluxstiffnessmatrix(nodes, nodes, dim, side) * jacobian(cell; ignore=dim)
  rev && reverse!(sfmm, dims=1) # to account for addition of flux terms from neighbours. It's complicated.
  @assert size(sfmm) == (nc, nc) "$(size(sfmm)) != ($nc, $nc)"
  fm = fluxmatrix(Val(dim), sfmm)
  @views output[1:3nc, 3nc+1:6nc] .-= fm .* speedoflight^2
  @views output[3nc+1:6nc, 1:3nc] .+= fm
  if !iszero(upwind)
    ufm = upwindfluxmatrix(Val(dim), sfmm) * upwind * jacobian(cell; ignore=dim)
    @views output[1:3nc, 1:3nc] .+= ufm .* epsilon0
    @views output[3nc+1:6nc, 3nc+1:6nc] .+= ufm
  end
  return output
end

function surfacefluxstiffnessmatrix(g::Grid{N,T}, upwind=0.0) where {N,T}
  output = spzeros(ndofs(g),ndofs(g))
  for cartindex in CartesianIndices(g.data)
    cellindex = Tuple(cartindex)
    cell = g[cartindex]
    nodes = NDimNodes(dofshape(cell), T)
    celldofindices = indices(g, cellindex)
    lumm = lu(kron(I(6), massmatrix(cell)))
    for dim in 1:N, (side, factor) in ((High, 1), (Low, -1))
      flux = surfacefluxstiffnessmatrix(cell, nodes, dim, side, false, upwind)
      @views output[celldofindices, celldofindices] .-= flux .* factor

      neighbourcellgridindex = findneighbourgridindex(g, cellindex, dim, side)
      neighbourcell = g[neighbourcellgridindex...]
      flux = surfacefluxstiffnessmatrix(neighbourcell, nodes, dim, opposite(side), true, upwind)
      ldiv!(lumm, flux)

      neighbourdofindices = indices(g, neighbourcellgridindex)
      @views output[celldofindices, neighbourdofindices] .+= flux .* factor
    end
    @views output[celldofindices, celldofindices] .= lumm \ output[celldofindices, celldofindices]
  end
  return output
end

function volumefluxstiffnessmatrix(cell::Cell{N}, nodes::NDimNodes) where {N}
  ns = ndofs(cell)
  output = spzeros(ns, ns)
  nc = ndofs(cell, 1) # number of dofs per component
  lumm = lu(massmatrix(nodes) * jacobian(cell))
  for dim in 1:N
    fmm = volumefluxstiffnessmatrix(nodes, nodes, dim) * jacobian(cell)
    ldiv!(lumm, fmm)
    fm = fluxmatrix(Val(dim), fmm)
    @views output[1:3nc, 3nc+1:6nc] .-= fm .* speedoflight^2
    @views output[3nc+1:6nc, 1:3nc] .+= fm
  end
  return output
end
volumefluxstiffnessmatrix(g::Grid{N,T}) where {N,T} = assembler(g, volumefluxstiffnessmatrix)

volumemassmatrix(c::Cell, n::NDimNodes) = kron(I(6), massmatrix(n)) * jacobian(c)
volumemassmatrix(g::Grid{N,T}) where {N,T} = assembler(g, volumemassmatrix)

function assembler(g::Grid{N,T}, f::F) where {N,T, F}
  n = ndofs(g)
  output = spzeros(n,n)
  for i in CartesianIndices(g.data)
    cellindices = indices(g, Tuple(i))
    nodes = NDimNodes(dofshape(g[i]), T)
    @views output[cellindices, cellindices] .+= f(g[i], nodes)
  end
  return output 
end

function assemble(g::Grid{N,T}; upwind=0.0) where {N, T}
  correctionfactor = numelements(g) / volume(g) * 2
  return correctionfactor .* (volumefluxstiffnessmatrix(g) + surfacefluxstiffnessmatrix(g, upwind))
end



