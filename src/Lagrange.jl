
abstract type BasisFunctionType end
abstract type AbstractLagrangeNodes{T} <: BasisFunctionType end
abstract type AbstractOrthogLagrangeNodes{T} <: AbstractLagrangeNodes{T} end

abstract type MaybeSolveDofs end
struct DelayDofsSolve <: MaybeSolveDofs end
struct SolveDofsNow <: MaybeSolveDofs end

opposite(_::DelayDofsSolve) = SolveDofsNow()
opposite(_::SolveDofsNow) = DelayDofsSolve()

for (stub, fname, orthog) in ((:Legendre, gausslegendre, true), (:Lobatto, gausslobatto, false))
  abstype = orthog ? :AbstractOrthogLagrangeNodes : :AbstractLagrangeNodes
  structname = Symbol(stub, :Nodes)
  @eval struct $(structname){T} <: $(abstype){T}
    nodes::Vector{T}
    weights::Vector{T}
    invdenominators::Vector{T}
  end
  @eval function $(structname)(N::Int)
    n, w = $(fname)(N)
    #n = (n .+ 1) ./ 2 .* (b - a) .+ a
    #w = w .* (b - a)/2
    invdenominators = [mapreduce(j->isequal(i, j) ? one(eltype(n)) : (1 / (n[i] - n[j])), *, eachindex(n)) for i in eachindex(n)]
    return $(structname)(n, w, invdenominators)
  end
end
function Base.isequal(a::AbstractLagrangeNodes, b::AbstractLagrangeNodes)
  length(a.nodes) == length(b.nodes) || return false
  for i in eachindex(a.nodes)
    a.nodes[i] == b.nodes[i] || return false
    a.weights[i] == b.weights[i] || return false
    a.invdenominators[i] == b.invdenominators[i] || return false
  end
  return true
end
Base.hash(n::AbstractLagrangeNodes) = foldr(hash, (n.nodes, n.nodes), init=hash(n.invdenominators))
node(n::AbstractLagrangeNodes, i::Int) = n.nodes[i]
weight(n::AbstractLagrangeNodes, i::Int) = n.weights[i]
nodes(n::AbstractLagrangeNodes) = n.nodes
weights(n::AbstractLagrangeNodes) = n.weights
Base.eltype(n::AbstractLagrangeNodes{T}) where {T<:Number} = T

@concrete struct NDimNodes
  nodes # Tuple of <:AbstractLagrangeNodes
end
Base.size(n::NDimNodes) = Tuple(length(n.nodes[i]) for i in eachindex(n))
Base.hash(n::NDimNodes) = mapreduce(hash, hash, n)

#_a(x) = -ones(length(x))
#_b(x) = ones(length(x))
#
#
_a(::Val{1}) = -1
_b(::Val{1}) = 1
_a(::Val{N}) where N = -(@SArray ones(N))
_b(::Val{N}) where N = @SArray ones(N)
_a(n::NDimNodes) = _a(Val(ndims(n)))
_b(n::NDimNodes) = _b(Val(ndims(n)))

function NDimNodes(sizetuple, ::Type{T}) where {T<:AbstractLagrangeNodes}
  N = length(sizetuple)
  return NDimNodes(NTuple{N,T}(T(sizetuple[i]) for i in eachindex(sizetuple)))
end

for fname in (:length, :iterate, :eachindex)
  @eval @inbounds Base.$(fname)(n::AbstractLagrangeNodes) = $(fname)(n.nodes)
  @eval @inbounds Base.$(fname)(n::NDimNodes) = $(fname)(n.nodes)
end
for fname in (:iterate, :getindex)
  @eval @inbounds Base.$(fname)(n::AbstractLagrangeNodes, i) = $(fname)(n.nodes, i)
  @eval @inbounds Base.$(fname)(n::NDimNodes, i) = $(fname)(n.nodes, i)
end

ndims(n::NDimNodes) = length(n.nodes)

#evaluate one function without coefficient
function (n::AbstractLagrangeNodes)(x::R, ind::Integer) where {R<:Real}
  T = promote_type(R, eltype(n))
  output = n.invdenominators[ind] * one(T)
  @inbounds for j in eachindex(n)
    j == ind && continue
    output *= (x - node(n, j))
  end
  return output
end

#evaluate one function without coefficient
lagrange(x::Number, nodes::AbstractLagrangeNodes, ind::Integer) = nodes(x, ind)
#function lagrange(x::Number, nodes::AbstractLagrangeNodes, ind::Integer)
#  T = promote_type(R, eltype(nodes))
#  output = one(T)
#  for j in eachindex(nodes)
#    j == ind && continue
#    output *= (x - nodes[j]) / (nodes[ind] - nodes[j])
#  end
#  return output
#end

#evaluate one function with coefficient
#lagrange(x::Real, nodes::AbstractLagrangeNodes, ind::Integer, coeff::Number) = lagrange(x, nodes, ind) * coeff
lagrange(x::Real, nodes::AbstractLagrangeNodes, ind::Integer, coeff::Number) = nodes(x, ind) * coeff

#evaluate one cartesian product of functions for one coefficient
function lagrange(x, nodes::NDimNodes, inds, coeff::T) where {T<:Number}
  #return prod(lagrange(x[i], nodes[i], inds[i]) for i in 1:length(x)) * coeff
  @assert length(x) == ndims(nodes)
  output = coeff
  @inbounds for i in 1:ndims(nodes)
    output *= (nodes[i])(x[i], inds[i])
  end
  return output
end

# evaluate all functions with all coefficients
function lagrange(x, nodes::NDimNodes, dofs)
  output = zero(promote_type(eltype(x), eltype(dofs)))
  for i in CartesianIndices(dofs)
    t = Tuple(i)
    output += lagrange(x, nodes, Tuple(i), dofs[i])
  end
  return output
end

maybesolvedofs!(dofs, nodes, a, b, _::DelayDofsSolve) = nothing
function maybesolvedofs!(dofs, nodes, a, b, _::SolveDofsNow)
  M = lumassmatrix(nodes, a, b)
  ldiv!((@view dofs[:]), M, (@view dofs[:]))
end

function solvedofsifnecessary!(dofs, nodes, a, b, d::DelayDofsSolve)
  maybesolvedofs!(dofs, nodes, a, b, opposite(d))
end

function lagrangeinner!(dofs, x::AbstractVector, nodes, value)
  for i in CartesianIndices(dofs)
    dofs[i] += lagrange(x, nodes, Tuple(i), value)
  end
end
function lagrange!(dofs, x::AbstractVector, nodes::NDimNodes, value,
    a=_a(nodes), b=_b(nodes), solvedofsornot::MaybeSolveDofs=SolveDofsNow())
  @assert size(dofs) == Tuple(length(n) for n in nodes)
  lagrangeinner!(dofs, x, nodes, value)
  maybesolvedofs!(dofs, nodes, a, b, solvedofsornot)
end
function lagrange!(dofs, xs::AbstractArray{<:Real, 2}, nodes::NDimNodes, value::T,
    a=_a(nodes), b=_b(nodes), solvedofsornot::MaybeSolveDofs=SolveDofsNow()) where {T}
  @assert size(dofs) == Tuple(length(n) for n in nodes)
  @assert size(xs, 1) == length(a)
  for i in 1:size(xs, 2)
    val = (T <: Number) ? value : value[i] # this isn't good but hopefully the compiler is cleverer than me
    lagrangeinner!(dofs, (@view xs[:, i]), nodes, val)
  end
  maybesolvedofs!(dofs, nodes, a, b, solvedofsornot)
end

quadrature(::Val{1}) = QuadGK.quadgk
quadrature(::Val{N}) where N = HCubature.hcubature
quadrature(n::NDimNodes) = quadrature(Val(ndims(n)))
function lagrange!(dofs, nodes::NDimNodes, f::F, a=_a(nodes), b=_b(nodes),
    solvedofsornot::MaybeSolveDofs=SolveDofsNow()) where {F<:Function}
  quad = quadrature(nodes)
  @assert size(dofs) == Tuple(length(n) for n in nodes)
  for i in CartesianIndices(dofs)
    t = Tuple(i)
    dofs[i] = quad(y->f(y) * lagrange(y, nodes, t, 1), a, b, atol=eps(), rtol=sqrt(eps()))[1]
  end
  maybesolvedofs!(dofs, nodes, a, b, solvedofsornot)
end
@memoize function massmatrix(nodesi::T, nodesj::U, a=_a(nodesi), b=_b(nodesi)
    ) where {T<:AbstractLagrangeNodes,U<:AbstractLagrangeNodes}
  @warn "Using non-orthogonal nodes $T, $U"
  output = zeros(length(nodesi), length(nodesj))
  for j in eachindex(nodesj), i in eachindex(nodesi)
    output[i, j] = QuadGK.quadgk(y->lagrange(y, nodesi, i, 1) * lagrange(y, nodesj, j, 1), a, b)[1]
  end
  return output
end
function massmatrix(nodesi::T, nodesj::T, a=_a(nodesi), b=_b(nodesi)
    ) where {U, T<:AbstractOrthogLagrangeNodes{U}}
#  @info "Calling memoized massmatrix(nodesi::T, nodesj::T, a=_a(nodesi), b=_b(nodesi))"
  return Diagonal(deepcopy(weights(nodesi)))
end

massmatrix(nodes::AbstractLagrangeNodes, a=_a(nodes), b=_b(nodes)) = massmatrix(nodes, nodes, a, b)

const massmatrixdictsdict = Dict{UInt64, Any}()

function massmatrixdictionary(nodesi::NDimNodes, nodesj::NDimNodes,
    a=_a(nodesi), b=_b(nodesi))
  key = mapreduce(hash, hash, (nodesi, nodesj, a, b))
  haskey(massmatrixdictsdict, key) && return massmatrixdictsdict[key]
#  throw(ErrorException("just show me the stack trace"))
  @info "Calling memoized massmatrixdictionary(...), $a, $b"
  @assert length(nodesi) == length(nodesj) == length(a) == length(b)
  szi = Tuple(length(n) for n in nodesi)
  szj = Tuple(length(n) for n in nodesj)
  M = zeros(prod(szi), prod(szj))
  Ms = [massmatrix(nodesi[i], nodesj[i], a[i], b[i]) for i in eachindex(nodesi)]
  @assert length(Ms) == length(nodesi)
  d = Dict()
  for i in CartesianIndices(szi), j in CartesianIndices(szj)
    indsi = Tuple(i)
    indsj = Tuple(j)
    d[(indsi, indsj)] = prod(Ms[k][indsi[k], indsj[k]] for k in 1:length(nodesi))
  end
  massmatrixdictsdict[key] = d
  return d
end
function massmatrixdictionary(nodes::NDimNodes, a=_a(nodes), b=_b(nodes))
  return massmatrixdictionary(nodes, nodes, a, b)
end


function massmatrix(nodesi::NDimNodes, nodesj::NDimNodes, a=_a(nodesi), b=_b(nodesi))
  szi = Tuple(length(n) for n in nodesi)
  szj = Tuple(length(n) for n in nodesj)
  M = zeros(prod(szi), prod(szj))
  d = massmatrixdictionary(nodesi, nodesj, a, b)
  for (j, jj) in enumerate(CartesianIndices(szj)), (i, ii) in enumerate(CartesianIndices(szi))
    indsi = Tuple(ii)
    indsj = Tuple(jj)
    M[i, j] = d[(indsi, indsj)]
  end
  return M
end
massmatrix(nodes::NDimNodes, a=_a(nodes), b=_b(nodes)) = massmatrix(nodes, nodes, a, b)
lumassmatrix(args...) = lu(massmatrix(args...))


const surfacefluxmassmatrixdict = Dict{UInt64, Any}()

surfacefluxmassmatrix(n::NDimNodes, dim::Int64, side::FaceDirection) = surfacefluxmassmatrix(n, n, dim, side)

function surfacefluxmassmatrix(nodesi::NDimNodes{1}, nodesj::NDimNodes{1}, dim::Int64, side::FaceDirection)
  @assert dim == 1
  key = mapreduce(hash, hash, (nodesi, nodesj, dim, side))
  haskey(surfacefluxmassmatrixdict, key) && return surfacefluxmassmatrixdict[key]
  x = side == Low ? -1 : 1
  szi = Tuple(length(n) for n in nodesi)
  szj = Tuple(length(n) for n in nodesj)
  output = zeros(prod(szi), prod(szj))
  for (j, jj) in enumerate(CartesianIndices(szj)), (i, ii) in enumerate(CartesianIndices(szi))
    output[i, j] = lagrange(x, nodesi[1], i) * lagrange(x, nodesj[1], j)
  end
  surfacefluxmassmatrixdict[key] = output
  return output
end

function surfacefluxmassmatrix(nodesi::NDimNodes, nodesj::NDimNodes, dim::Int64, side::FaceDirection)
  N = ndims(nodesi)
  @assert 0 < dim <= N
  @assert N == ndims(nodesj)

  key = mapreduce(hash, hash, (nodesi, nodesj, dim, side))
  haskey(surfacefluxmassmatrixdict, key) && return surfacefluxmassmatrixdict[key]

  szi = Tuple(length(n) for n in nodesi)
  szj = Tuple(length(n) for n in nodesj)
  output = ones(prod(szi), prod(szj))
  for (j, jj) in enumerate(CartesianIndices(szj)), (i, ii) in enumerate(CartesianIndices(szi))
    for d in 1:N # integrating over face at *side* of cell at const *dim*
      indi, indj = Tuple(ii)[d], Tuple(jj)[d]
      kernel(x) = lagrange(x, nodesi[d], indi) * lagrange(x, nodesj[d], indj)
      if d == dim
        x = side == Low ? -1.0 : 1.0
        output[i,j] *= kernel(x)
      else
        output[i,j] *= quadgk(kernel, -1, 1, rtol=eps())[1]
      end
    end
  end
  surfacefluxmassmatrixdict[key] = output
  return output
end


const volumefluxmassmatrixdict = Dict{UInt64, Any}()

function volumefluxmassmatrix(nodesi::NDimNodes, nodesj::NDimNodes, dim::Int)
  N = ndims(nodesi)
  @assert 0 < dim <= ndims(nodesi)
  @assert N == ndims(nodesj)

  key = mapreduce(hash, hash, (nodesi, nodesj, dim))
  haskey(volumefluxmassmatrixdict, key) && return volumefluxmassmatrixdict[key]

  output = ones(length(nodesi), length(nodesj))
  szi = Tuple(length(n) for n in nodesi)
  szj = Tuple(length(n) for n in nodesj)
  output = ones(prod(szi), prod(szj))
  #quad = quadrature(Val(N))
  for d in 1:N
    for (j, jj) in enumerate(CartesianIndices(szj)), (i, ii) in enumerate(CartesianIndices(szi))
      indi, indj = Tuple(ii)[d], Tuple(jj)[d]
      if d == dim
        output[i, j] *= QuadGK.quadgk(y->lagrange(y, nodesi[d], indi) * lagrangederiv(y, nodesj[d], indj), -1, 1)[1]
      else
        output[i, j] *= QuadGK.quadgk(y->lagrange(y, nodesi[d], indi) * lagrange(y, nodesj[d], indj), -1, 1)[1]
      end
    end
  end
  volumefluxmassmatrixdict[key] = output
  return output
end


function convertbases!(dofsto, dofsfrom, nodesfrom, nodesto, a=_a(nodesfrom), b=_b(nodesfrom))
  @assert size(dofsto) == size(dofsfrom)
  Mtoto = massmatrix(nodesto, nodesto, a, b)
  Mtofrom = massmatrix(nodesto, nodesfrom, a, b)
  @views dofsto[:] .= Mtoto \ Mtofrom * dofsfrom[:]
end


# derivative of the ind^th nodal lagrange function associated with nodes at position x
function lagrangederiv(x::R, nodes::AbstractLagrangeNodes{T}, ind::Integer) where {R<:Real, T}
  return sum(i-> 1 / (nodes[ind] - nodes[i]) *
    mapreduce(j->(x - nodes[j]) / (nodes[ind] - nodes[j]), *,
              filter(j->!(j in (ind, i)), eachindex(nodes)); init=one(promote_type(R, T))),
    filter(i->!(i in ind), eachindex(nodes)))
end


# evaluate derivative along the direction, dirdiection, of the nodal lagrange function
# product associated with `coeff` for `nodes` at position x
function lagrangederiv(x, nodes::NDimNodes, inds, coeff::Number, derivdirection::Integer)
  output = one(promote_type(eltype(x), typeof(coeff))) * coeff
  for i in 1:length(x)
    if derivdirection == i
      output *= (lagrangederiv(x[i], nodes[i], inds[i]) for i in 1:length(x))
    else
      output *= (lagrange(x[i], nodes[i], inds[i]) for i in 1:length(x))
    end
  end
  return output
end

function lagrangederiv(x, nodes::NDimNodes, dofs::AbstractArray, derivdirection::Integer)
  return mapreduce(i->lagrangederiv(x, nodes, Tuple(i), dofs[i], derivdirection), +, CartesianIndices(dofs))
end


