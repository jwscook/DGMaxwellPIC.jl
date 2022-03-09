
@memoize function volumemassmatrix(sizetuple, ::Type{LagrangeOrthogonal})
  N = length(sizetuple)
  v = zeros(prod(sizetuple)) # we know LagrangeOrthogonal polynomials are orthogonal, so only need a vector
  weights = gausslegendreweights(sizetuple)
  for (i, c) in enumerate(CartesianIndices(sizetuple))
    tc = Tuple(c)
    v[i] = mapreduce(j->weights[j][tc[j]], *, 1:N)
  end
  return Diagonal(v) # turn into a Diagonal matrix
end

@memoize function integrate(nodes::Vector{T}, i, j, ::Type{LagrangeOrthogonal}, jderivative::Bool=false)::T where {T}
  if jderivative
    output = quadgk(x->lagrange(x, nodes, i) * lagrangederiv(x, nodes, j), -1, 1, rtol=eps(T))[1]
    return abs(output) <= eps(T) ? zero(T) : output
  else
    weights = gausslegendre(length(nodes))[2] 
    return (i == j) * weights[i]
  end
end
@memoize function stiffnessmatrix(sizetuple, ::Type{<:Lagrange}, derivativedirection::Integer)
  nodesvector = gausslegendrenodes(sizetuple)
  N = length(sizetuple)
  function integrand(directionij)
    direction, i, j = directionij
    nodes = nodesvector[direction]
    return integrate(nodes, i, j, LagrangeOrthogonal, direction == derivativedirection)
  end
  output = zeros(prod(sizetuple), prod(sizetuple))
  cart = CartesianIndices(sizetuple)
  for (i, c) in enumerate(cart), (j, k) in enumerate(cart)
    output[i, j] = mapreduce(integrand, *, zip(1:N, Tuple(c), Tuple(k)))
  end
  return output 
end

@memoize function facemassmatrix(sizetuple, dim::Int64, side::FaceDirection, ::Type{LagrangeOrthogonal})
  N = length(sizetuple)
  output = zeros(prod(sizetuple)) # we know LagrangeOrthogonal polynomials are orthogonal, so only need a vector
  weights = gausslegendreweights(sizetuple)
  fdi = facedofindices(sizetuple, dim, side)
  cart = CartesianIndices(sizetuple)
  for (i, g) in enumerate(fdi)
    tc = Tuple(cart[g])
    output[g] = prod(j == dim ? 1 : weights[j][tc[j]] for j in 1:N) # integrate
  end
  return Diagonal(output) # turn into a Diagonal matrix
end

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

@memoize function surfacefluxmassmatrix(cellorstate::V, dim, side::FaceDirection, upwind=0.0
    ) where {N,T,V<:Union{Cell{N,T},State{N,T}}}
  output = spzeros(ndofs(cellorstate), ndofs(cellorstate))
  return surfacefluxmassmatrix!(output, cellorstate, dim, side, upwind)
end
@memoize function surfacefluxmassmatrix!(output, cellorstate::V, dim, side::FaceDirection, upwind=0.0
    ) where {N,T,V<:Union{Cell{N,T},State{N,T}}}
  @assert size(output) == (ndofs(cellorstate), ndofs(cellorstate))
  nc = ndofs(cellorstate, 1) # number of dofs per component
  fm = fluxmatrix(Val(dim), spdiagm(sparsevec(facedofindices(cellorstate, dim, side), true, nc)))
  @views output[1:3nc, 3nc+1:6nc] .+= fm .* speedoflight^2
  @views output[3nc+1:6nc, 1:3nc] .-= fm
  if !iszero(upwind)
    fm = upwindfluxmatrix(Val(dim), spdiagm(sparsevec(facedofindices(cellorstate, dim, side), true, nc))) .* speedoflight * upwind / 2
    @views output[1:3nc, 1:3nc] .+= fm .* epsilon0
    @views output[3nc+1:6nc, 3nc+1:6nc] .-= fm
  end
  return output
end

@memoize function surfacefluxmassmatrix!(dict::Dict, cellorstate::V, upwind=0.0) where {N,T,V<:Union{Cell{N,T},State{N,T}}}
  for dim in 1:N, side in (Low, High)
    dict[(dim, side)] = surfacefluxmassmatrix(cellorstate, dim, side, upwind)
  end
  return dict
end
@memoize function surfacefluxmassmatrix(cellorstate::V, upwind=0.0) where {N,T,V<:Union{Cell{N,T},State{N,T}}}
  ns = ndofs(cellorstate)
  output = spzeros(ns, ns)
  for dim in 1:N, side in (Low, High)
    surfacefluxmassmatrix!(output, cellorstate, dim, side, upwind)
  end
  return output
end

@memoize function volumefluxmassmatrix(cellorstate::V) where {N,T,V<:Union{Cell{N,T},State{N,T}}}
  ns = ndofs(cellorstate)
  output = spzeros(ns, ns) # we know lagrangeorthogonal polynomials are orthogonal, so only need a vector
  nc = ndofs(cellorstate, 1) # number of dofs per component
  for dim in 1:N
    fm = fluxmatrix(Val(dim), nc)
    @views output[1:3nc, 3nc+1:6nc] .+= fm .* speedoflight^2
    @views output[3nc+1:6nc, 1:3nc] .-= fm
  end
  return output
end

function volumemassmatrix(cellorstate::V) where {N,T,V<:Union{Cell{N,T},State{N,T}}}
  return kron(I(6), volumemassmatrix(dofshape(cellorstate), T))
end


function assembler(g::Grid{N,T}, f::F) where {N,T, F}
  n = ndofs(g)
  output = spzeros(n,n)
  for i in CartesianIndices(g.data)
    cellindices = indices(g, Tuple(i))
    @views output[cellindices, cellindices] .= f(g[i])
  end
  return output 
end
function volumefluxmassmatrix(g::Grid{N,T}) where {N,T}
  return assembler(g, volumefluxmassmatrix)
end

function volumemassmatrix(g::Grid{N,T}) where {N,T}
  return assembler(g, volumemassmatrix)
end

@memoize function indices(g::Grid{N,T}, cellindices, fndofs::F=ndofs) where {N,T,F}
  @assert all(ones(N) .<= cellindices .<= size(g.data))
  offset = 0
  for i in CartesianIndices(g.data)
    if all(j->isequal(j...), zip(cellindices, Tuple(i)))
      return (1:fndofs(g[i])) .+ offset
    end
    offset += fndofs(g[i])
  end
  throw(ErrorException("Shouldn't be able to get here"))
end

function surfacefluxmassmatrix(g::Grid{N,T}, upwind=0.0) where {N,T}
  gridsize = size(g)
  n = ndofs(g)
  output = spzeros(n,n)
  for cartindex in CartesianIndices(g.data)
    cellindex = Tuple(cartindex)
    cell = g[cartindex]
    celldofindices = indices(g, cellindex)
    for dim in 1:N, side in (Low, High)
      neighbour = [mod1(cellindex[j] + (dim == j) * ((side == Low) ? -1 : 1), gridsize[j]) for j in 1:N]
      neighbourdofindices = indices(g, neighbour)
      flux = surfacefluxmassmatrix(cell, dim, side, upwind)
      @views output[neighbourdofindices, celldofindices] .+= flux
      @views output[celldofindices, celldofindices] .+= flux
    end
  end
  return output 
end

invvolumemassmatrix(g::Grid{N,LagrangeOrthogonal}) where N = Diagonal(1 ./ diag(volumemassmatrix(g)))

function assemble(g::Grid, upwind=0.0)
  return invvolumemassmatrix(g) * (volumefluxmassmatrix(g) .+ surfacefluxmassmatrix(g, upwind))
end



