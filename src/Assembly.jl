
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

# positive curl / levi-civita
fluxmatrix(::Val{1}, x) = _fluxmatrix((@SArray [0 0 0; 0 0 1; 0 -1 0]), x)
fluxmatrix(::Val{2}, x) = _fluxmatrix((@SArray [0 0 -1; 0 0 0; 1 0 0]), x)
fluxmatrix(::Val{3}, x) = _fluxmatrix((@SArray [0 1 0; -1 0 0; 0 0 0]), x)

function surfacefluxstiffnessmatrix(nodesi::NDimNodes, nodesj::NDimNodes, sidei::FaceDirection,
    sidej::FaceDirection, dim::Integer, upwind=0.0)
  output = zeros(6ndofs(nodesi), 6ndofs(nodesj))
  return surfacefluxstiffnessmatrix!(output, nodesi, nodesj, sidei, sidej, dim, upwind)
end

function surfacefluxstiffnessmatrix!(output, nodesi::NDimNodes, nodesj::NDimNodes, sidei::FaceDirection, sidej::FaceDirection, dim::Integer, upwind=0.0)
  nci = ndofs(nodesi) # number of dofs per component
  ncj = ndofs(nodesj) # number of dofs per component
  sfmm = surfacefluxstiffnessmatrix(nodesi, nodesj, sidei, sidej, dim)
  @assert size(sfmm) == (nci, ncj) "$(size(sfmm)) != ($nci, $ncj)"
  fm = fluxmatrix(Val(dim), sfmm)
  @views output[1:3nci, 3ncj+1:6ncj] .-= fm .* speedoflight^2
  @views output[3nci+1:6nci, 1:3ncj] .+= fm
  if !iszero(upwind)
    ufm = upwindfluxmatrix(Val(dim), sfmm) * upwind * speedoflight / 2
    sidei == Low && (ufm .*= -1)
    @views output[1:3nci, 1:3ncj] .+= ufm .* epsilon0
    @views output[3nci+1:6nci, 3ncj+1:6ncj] .+= ufm
  end
  return output
end

function surfacefluxstiffnessmatrix(g::Grid{N,T}, upwind=0.0) where {N,T}
  output = spzeros(ndofs(g),ndofs(g))
  for cartindex in CartesianIndices(g.data)
    cellindex = Tuple(cartindex)
    cell = g[cartindex]
    nodesi = NDimNodes(dofshape(cell), T)
    celldofindices = indices(g, cellindex)
    lumm = lu(kron(I(6), massmatrix(cell)))
    for dim in 1:N, (side, factor) in ((High, 1), (Low, -1))
      flux = surfacefluxstiffnessmatrix(nodesi, nodesi, side, side, dim, upwind)
      flux .*= jacobian(cell; ignore=dim)
      @views output[celldofindices, celldofindices] .-= flux .* factor

      neighbourcellgridindex = findneighbourgridindex(g, cellindex, dim, side)
      neighbourcell = g[neighbourcellgridindex...]
      nodesj = NDimNodes(dofshape(neighbourcell), T)
      flux = surfacefluxstiffnessmatrix(nodesi, nodesj, side, opposite(side), dim, upwind)
      flux .*= jacobian(cell; ignore=dim)
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
  output = zeros(ns, ns)
  nc = ndofs(cell, 1) # number of dofs per component
  lumm = lu(kron(I(6), massmatrix(cell) * jacobian(cell)))
  for dim in 1:N
    fmm = volumefluxstiffnessmatrix(nodes, nodes, dim) * jacobian(cell)
    fm = Matrix(fluxmatrix(Val(dim), fmm))
    @views output[1:3nc, 3nc+1:6nc] .-= fm .* speedoflight^2
    @views output[3nc+1:6nc, 1:3nc] .+= fm
  end
  ldiv!(lumm, output)
  return sparse(output)
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
  return volumefluxstiffnessmatrix(g) + surfacefluxstiffnessmatrix(g, upwind)
end



