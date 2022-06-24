
struct SparseIJV{T}
  I::Vector{Int}
  J::Vector{Int}
  V::Vector{T}
  index::Ref{Int}
  length::Ref{Int}
end
SparseIJV(I, J, V) = SparseIJV(I, J, V, Ref(0), Ref(length(I)))
SparseIJV(n::Int, ::Type{T}) where  T = SparseIJV(zeros(Int, n), zeros(Int, n), zeros(T, n))
function Base.setindex!(s::SparseIJV{T}, ij::Tuple, v::T) where {T}
  s.index[] += 1
  if s.index[] > s.length[]
    s.length[] *= 2
    resize!(s.I, s.length[])
    resize!(s.J, s.length[])
    resize!(s.V, s.length[])
  end
  s.I[s.index[]] = ij[1]
  s.J[s.index[]] = ij[2]
  s.V[s.index[]] = v
end
function Base.setindex!(s::SparseIJV{T}, is, js, vs::AbstractArray{T}) where {T}
  for (c, ij) in enumerate(CartesianIndices((is, js)))
    setindex!(s, Tuple(ij), vs[c])
  end
end
function zero!(s::SparseIJV)
  s.index[] = 0
  fill!(s.I, 0)
  fill!(s.J, 0)
  fill!(s.V, 0)
end

function findneighbourgridindex(g::Grid{N}, homeindex, searchdir, side) where N
  increment = MVector{N,Int}(zeros(N))
  increment[searchdir] += (side == Low) ? -1 : 1
  return SVector{N,Int}(mod1.(homeindex .+ increment, size(g)))
end
function findneighbourcell(g::Grid{N}, homeindex, searchdir, side) where N
  return g[findneighbourgridindex(g, homeindex, searchdir, side)]
end

_fluxmatrix(stencil, x::AbstractArray) = sparse(kron(stencil, x))
upwindfluxmatrix(::Val{1}, x) = _fluxmatrix((@SArray [0 0 0; 0 1 0; 0 0 1]), x)
upwindfluxmatrix(::Val{2}, x) = _fluxmatrix((@SArray [1 0 0; 0 0 0; 0 0 1]), x)
upwindfluxmatrix(::Val{3}, x) = _fluxmatrix((@SArray [1 0 0; 0 1 0; 0 0 0]), x)

# positive curl / levi-civita
fluxmatrix(::Val{1}, x) = _fluxmatrix((@SArray [0 0 0; 0 0 1; 0 -1 0]), x)
fluxmatrix(::Val{2}, x) = _fluxmatrix((@SArray [0 0 -1; 0 0 0; 1 0 0]), x)
fluxmatrix(::Val{3}, x) = _fluxmatrix((@SArray [0 1 0; -1 0 0; 0 0 0]), x)

function surfacefluxstiffnessmatrix!(output, nodesi::NDimNodes, nodesj::NDimNodes,
    sidei::FaceDirection, sidej::FaceDirection, dim::Integer, upwind=0.0)
  fill!(output, 0)
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

function surfacefluxstiffnessmatrix!(ijv::SparseIJV, g::Grid{N,T}, cellindex::Tuple,
    upwind=0.0) where {N,T}
  zero!(ijv)
  cell = g[cellindex]
  nodesi = NDimNodes(dofshape(cell), T)
  celldofindices = indices(g, cellindex)
  lumm = lumassmatrix(g, cell)

  fluxii = zeros(6ndofs(nodesi), 6ndofs(nodesi))

  for dim in 1:N, (side, factor) in ((High, 1), (Low, -1))
    surfacefluxstiffnessmatrix!(fluxii, nodesi, nodesi, side, side, dim, upwind)
    fluxii .*= -jacobian(cell; ignore=dim) * factor
    ldiv!(lumm, fluxii)
    setindex!(ijv, celldofindices, celldofindices, fluxii)

    neighbourcellgridindex = findneighbourgridindex(g, cellindex, dim, side)
    neighbourcell = g[neighbourcellgridindex]
    nodesj = NDimNodes(dofshape(neighbourcell), T)
    fluxij = ndofs(nodesi) == ndofs(nodesj) ? fluxii : zeros(6ndofs(nodesi), 6ndofs(nodesj))
    surfacefluxstiffnessmatrix!(fluxij, nodesi, nodesj, side, opposite(side), dim, upwind)
    fluxij .*= jacobian(cell; ignore=dim) * factor
    ldiv!(lumm, fluxij)

    neighbourdofindices = indices(g, neighbourcellgridindex)
    setindex!(ijv, celldofindices, neighbourdofindices, fluxij)
  end
end

function surfacefluxstiffnessmatrix!(output, g::Grid{N,T}, upwind=0.0) where {N,T}
  nnzs = (Int(round(ndofs(g) / length(g))))^2 * 5N # dumb & likely massive over-estimate of nonzeros
  ijv = SparseIJV(nnzs, Float64)
  for cartindex in CartesianIndices(g.data)
    surfacefluxstiffnessmatrix!(ijv, g, Tuple(cartindex), upwind)
    term = sparse((@view ijv.I[1:ijv.index[]]),
                  (@view ijv.J[1:ijv.index[]]),
                  (@view ijv.V[1:ijv.index[]]),
                  ndofs(g), ndofs(g))
    output .+= term
  end
  return output
end

function _surfacefluxstiffnessmatrix!(output, g::Grid{N,T}, upwind=0.0) where {N,T}
  for cartindex in CartesianIndices(g.data)
    cellindex = Tuple(cartindex)
    cell = g[cartindex]
    nodesi = NDimNodes(dofshape(cell), T)
    flux = zeros(6ndofs(nodesi), 6ndofs(nodesi))
    celldofindices = indices(g, cellindex)
    lumm = lumassmatrix(g, cell)
    for dim in 1:N, (side, factor) in ((High, 1), (Low, -1))
      surfacefluxstiffnessmatrix!(flux, nodesi, nodesi, side, side, dim, upwind)
      flux .*= jacobian(cell; ignore=dim)
      ldiv!(lumm, flux)
      @views output[celldofindices, celldofindices] .-= flux .* factor

      neighbourcellgridindex = findneighbourgridindex(g, cellindex, dim, side)
      neighbourcell = g[neighbourcellgridindex]
      nodesj = NDimNodes(dofshape(neighbourcell), T)
      surfacefluxstiffnessmatrix!(flux, nodesi, nodesj, side, opposite(side), dim, upwind)
      flux .*= jacobian(cell; ignore=dim)
      ldiv!(lumm, flux)
      @views output[celldofindices, indices(g, neighbourcellgridindex)] .+= flux .* factor
    end
  end
  return output
end

function volumefluxstiffnessmatrix(cell::Cell{N,T}, lumm) where {N,T}
  nodes = NDimNodes(dofshape(cell), T)
  output = zeros(ndofs(cell), ndofs(cell))
  nc = ndofs(cell, 1) # number of dofs per component
  for dim in 1:N
    fmm = volumefluxstiffnessmatrix(nodes, nodes, dim) * jacobian(cell; ignore=dim)
    fm = fluxmatrix(Val(dim), fmm)
    @views output[1:3nc, 3nc+1:6nc] .-= fm .* speedoflight^2
    @views output[3nc+1:6nc, 1:3nc] .+= fm
  end
  ldiv!(lumm, output)
  return sparse(output)
end

function volumefluxstiffnessmatrix!(output, g::Grid{N,T}) where {N,T}
  for i in CartesianIndices(g.data)
    cellindices = indices(g, Tuple(i))
    cell = g[i]
    lumm = lumassmatrix(g, cell)
    @views output[cellindices, cellindices] .+= volumefluxstiffnessmatrix(cell, lumm)
  end
  return output 
end

function assemble(g::Grid{N,T}; upwind=0.0) where {N, T}
  n = ndofs(g)
  output = spzeros(n,n)
  volumefluxstiffnessmatrix!(output, g)
  surfacefluxstiffnessmatrix!(output, g, upwind)
  return output
#  I, J = SparseArrays.findnz(output)
#  V = SparseArrays.nonzeros(output)
#  return sparsecsr(I, J, V) # sparsecsr can use threading
end

@inline function currentloadvector!(output, g::Grid{N, T}, cellindex) where {N,T}
  @inbounds cell = g[cellindex]
  nc = ndofs(cell, 1) # number of dofs per component
  @assert length(output) == 6nc "$(length(output)) vs $(6nc)"
  @inbounds @views output[1:3nc] .= currentdofs(cell) # ∇×B = μJ + μϵ ∂E/∂t # yes electric current
  #No-op #@views output[3nc+1:6nc] .= 0 # ∂B/∂t = - ∇×E # no magnetic current
end


function currentloadvector!(output, g::Grid{N,T}) where {N,T}
  @threads for i in CartesianIndices(g.data)
    cellindices = indices(g, Tuple(i))
    currentloadvector!((@view output[cellindices]), g, i)
  end
  return output
end

function currentsource!(output, g::Grid)
  return currentloadvector!(output, g)
end


