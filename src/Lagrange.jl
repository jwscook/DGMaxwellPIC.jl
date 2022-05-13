
abstract type BasisFunctionType end
abstract type AbstractLagrangeNodes <: BasisFunctionType end
abstract type AbstractOrthogLagrangeNodes <: AbstractLagrangeNodes end

abstract type MaybeSolveDofs end
struct DelayDofsSolve <: MaybeSolveDofs end
struct SolveDofsNow <: MaybeSolveDofs end

opposite(_::DelayDofsSolve) = SolveDofsNow()
opposite(_::SolveDofsNow) = DelayDofsSolve()

for (stub, fname, orthog) in ((:Legendre, gausslegendre, true), (:Lobatto, gausslobatto, false))
  abstype = orthog ? :AbstractOrthogLagrangeNodes : :AbstractLagrangeNodes
  structname = Symbol(stub, :Nodes)
  @eval struct $(structname) <: $(abstype)
    nodes::Vector{Float64}
    weights::Vector{Float64}
    invdenominators::Vector{Float64}
    uniqueid::UInt64
  end
  @eval function $(structname)(N::Int)
    n, w = $(fname)(N)
    invdenominators = [mapreduce(j->isequal(i, j) ? one(eltype(n)) : (1 / (n[i] - n[j])), *, eachindex(n)) for i in eachindex(n)]
    uniqueid = mapreduce(hash, hash, (n, w, invdenominators))
    return $(structname)(n, w, invdenominators, uniqueid)
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
uniqueid(n::AbstractLagrangeNodes) = n.uniqueid
Base.eltype(n::AbstractLagrangeNodes) = Float64

struct NDimNodes{N,T<:AbstractLagrangeNodes}
  nodes::NTuple{N,T} # Tuple of <:AbstractLagrangeNodes
  uniqueid::UInt64
  works::Vector{Array{Float64, N}}
  lumm::LU{Float64, Matrix{Float64}}
  function NDimNodes(nodes::NTuple{N,T}) where {N,T}
    work = zeros(Float64, (length(n) for n in nodes)...)
    works = [deepcopy(work) for i in 1:nthreads()]
    fakelu = lu(rand(1,1))
    uniqueid = mapreduce(hash, hash, nodes)
    fakenew = new{N,T}(nodes, uniqueid, works, lu(ones(1,1)))
    lumm = lu(massmatrix(fakenew, fakenew))
    return new{N,T}(nodes, uniqueid, works, lumm)
  end
end
Base.size(n::NDimNodes) = Tuple(length(n.nodes[i]) for i in eachindex(n))
Base.hash(n::NDimNodes) = n.uniqueid
ndofs(n::NDimNodes) = prod(length, n.nodes)
workarray(n::NDimNodes, tid) = n.works[tid]

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
function (n::AbstractLagrangeNodes)(x::R, ind::Integer) where {R<:Number}
  T = promote_type(R, eltype(n))
  output = T(n.invdenominators[ind])
  @inbounds for j in eachindex(n)
    output *= j == ind ? T(1) : T(x - node(n, j))
  end
  return output::T
end

lagrange(x::Number, n::AbstractLagrangeNodes, ind::Integer) = n(x, ind)
lagrange(x::Number, n::AbstractLagrangeNodes, ind::Integer, coeff::Number) = n(x, ind) * coeff

function lagrange(x, nodes::NDimNodes, inds::NTuple{N,Integer}) where {N}
  @assert length(x) == ndims(nodes) == N
  output = nodes[1](x[1], inds[1])
  for i in 2:ndims(nodes)
    output *= nodes[i](x[i], inds[i])
  end
  return output
end

function lagrange(x, nodes::NDimNodes{N}, dofsargs::Vararg{T,M}) where {N,T,M}
  T1 = promote_type(eltype(x), eltype(first(dofsargs)))
  output = MVector{M,T1}(undef)
  lagrange!(output, x, nodes, dofsargs...)
  return output
end
function lagrange!(output, x, nodes::NDimNodes{N}, dofsargs::Vararg{T,M}) where {N,T,M}
  work = workarray(nodes, threadid())
  @inbounds for i in CartesianIndices(work)
    work[i] = lagrange(x, nodes, Tuple(i))
  end
  @inbounds for (j, dofs) in enumerate(dofsargs)
    output[j] = dot(work, dofs)
  end
  return output
end


maybesolvedofs!(dofs, nodes, _::DelayDofsSolve) = nothing
function maybesolvedofs!(dofs, nodes, _::SolveDofsNow)
  M = lumassmatrix(nodes)
  ldiv!((@view dofs[:]), M, (@view dofs[:]))
end

function solvedofsifnecessary!(dofs, nodes, d::DelayDofsSolve)
  maybesolvedofs!(dofs, nodes, opposite(d))
end

function lagrangeinner!(dofs, x::AbstractVector, nodes, value)
  @inbounds for i in CartesianIndices(dofs)
    dofs[i] += lagrange(x, nodes, Tuple(i)) * value
  end
end
function lagrange!(dofs, x::AbstractVector, nodes::NDimNodes{N}, value,
    solvedofsornot::MaybeSolveDofs) where N
  @assert all(i->size(dofs, i) == length(nodes[i]), 1:N)
  lagrangeinner!(dofs, x, nodes, value)
  maybesolvedofs!(dofs, nodes, solvedofsornot)
end

function lagrange!(dofs, xs::AbstractArray{<:Real, 2}, nodes::NDimNodes{N}, value::T,
    solvedofsornot::MaybeSolveDofs) where {N,T}
  @assert all(i->size(dofs, i) == length(nodes[i]), 1:N)
  @inbounds @views for i in axes(xs, 2)
    val = (T <: Number) ? value : value[i] # this isn't good but the compiler can figure it out
    lagrangeinner!(dofs, xs[:, i], nodes, val)
  end
  maybesolvedofs!(dofs, nodes, solvedofsornot)
end

function lagrange!(dofs, nodes::NDimNodes, f::F,
    solvedofsornot::MaybeSolveDofs=SolveDofsNow()) where {F<:Function}
  a, b = _a(nodes), _b(nodes)
  @assert size(dofs) == Tuple(length(n) for n in nodes)
  @inbounds for i in CartesianIndices(dofs)
    dofs[i] = HCubature.hcubature(x->f(x) * lagrange(x, nodes, Tuple(i)), a, b,
                                  atol=ATOL, rtol=sqrt(eps()))[1]
    #dofs[i] *= abs(dofs[i] > 100ATOL)
  end
  maybesolvedofs!(dofs, nodes, solvedofsornot)
end
function massmatrix(nodesi::T, nodesj::U,
    ) where {T<:AbstractLagrangeNodes,U<:AbstractLagrangeNodes}
  output = zeros(length(nodesi), length(nodesj))
  @inbounds for j in eachindex(nodesj), i in eachindex(nodesi)
    output[i, j] = QuadGK.quadgk(x->lagrange(x, nodesi, i) * lagrange(x, nodesj, j),
                                 -1, 1, atol=ATOL, rtol=RTOL)[1]
    #output[i, j] *= abs(output[i, j] > 100ATOL)
  end
  return output
end
function massmatrix(nodesi::T, nodesj::T
    ) where {T<:AbstractOrthogLagrangeNodes}
  return Diagonal(deepcopy(weights(nodesi)))
end

massmatrix(n::AbstractLagrangeNodes) = massmatrix(n, n)
massmatrix(n::AbstractLagrangeNodes, jacobian::Number) = massmatrix(n, n) * jacobian

const massmatrixdictsdict = Dict{UInt64, Any}()

function massmatrixdictionary(nodesi::NDimNodes, nodesj::NDimNodes)
  key = mapreduce(hash, hash, (nodesi, nodesj))
  haskey(massmatrixdictsdict, key) && return massmatrixdictsdict[key]
  @assert length(nodesi) == length(nodesj)
  szi = Tuple(length(n) for n in nodesi)
  szj = Tuple(length(n) for n in nodesj)
  M = zeros(prod(szi), prod(szj))
  Ms = [massmatrix(nodesi[i], nodesj[i]) for i in eachindex(nodesi)]
  @assert length(Ms) == length(nodesi)
  d = Dict()
  @inbounds for i in CartesianIndices(szi), j in CartesianIndices(szj)
    indsi = Tuple(i)
    indsj = Tuple(j)
    d[(indsi, indsj)] = prod(Ms[k][indsi[k], indsj[k]] for k in 1:length(nodesi))
  end
  massmatrixdictsdict[key] = d
  return d
end
massmatrixdictionary(nodes::NDimNodes) = massmatrixdictionary(nodes, nodes)

const _massmatrixdict = Dict{UInt64, Any}()

function massmatrix(nodesi::NDimNodes, nodesj::NDimNodes)
  key = mapreduce(hash, hash, (nodesi, nodesj))
  haskey(_massmatrixdict, key) && return _massmatrixdict[key]

  szi = Tuple(length(n) for n in nodesi)
  szj = Tuple(length(n) for n in nodesj)
  M = zeros(prod(szi), prod(szj))
  d = massmatrixdictionary(nodesi, nodesj)
  for (j, jj) in enumerate(CartesianIndices(szj)), (i, ii) in enumerate(CartesianIndices(szi))
    indsi = Tuple(ii)
    indsj = Tuple(jj)
    M[i, j] = d[(indsi, indsj)]
  end
  _massmatrixdict[key] = M
  return _massmatrixdict[key]
end

massmatrix(nodes::NDimNodes) = massmatrix(nodes, nodes)

lumassmatrix(n::NDimNodes) = n.lumm

const _lumassmatrixdict = Dict{UInt64, Any}()
function lumassmatrix(args...)
  key = mapreduce(hash, hash, args)
  #haskey(_lumassmatrixdict, key) && return _lumassmatrixdict[key]

  output = lu(massmatrix(args...))
  #_lumassmatrixdict[key] = output
  return output
end


const surfacefluxstiffnessmatrixdict = Dict{UInt64, Any}()

function surfacefluxstiffnessmatrix(nodesi::NDimNodes, nodesj::NDimNodes, sidei::FaceDirection, sidej::FaceDirection, dim::Int64)
  N = ndims(nodesi)
  @assert 0 < dim <= N
  @assert N == ndims(nodesj)

  key = mapreduce(hash, hash, (nodesi, nodesj, sidei, sidej, dim))
  haskey(surfacefluxstiffnessmatrixdict, key) && return surfacefluxstiffnessmatrixdict[key]

  szi = Tuple(length(n) for n in nodesi)
  szj = Tuple(length(n) for n in nodesj)
  output = ones(prod(szi), prod(szj))
  for (j, jj) in enumerate(CartesianIndices(szj)), (i, ii) in enumerate(CartesianIndices(szi))
    for d in 1:N # integrating over face at *side* of cell at const *dim*
      indi, indj = Tuple(ii)[d], Tuple(jj)[d]
      if d == dim
        xi = sidei == Low ? -1.0 : 1.0
        xj = sidej == Low ? -1.0 : 1.0
        output[i,j] *= lagrange(xi, nodesi[d], indi) * lagrange(xj, nodesj[d], indj)
      else
        kernel(x) = lagrange(x, nodesi[d], indi) * lagrange(x, nodesj[d], indj)
        integral = QuadGK.quadgk(kernel, -1, 1, atol=ATOL, rtol=RTOL)[1]
        #integral *= abs(integral > 100ATOL)
        output[i,j] *= integral
      end
    end
  end
  surfacefluxstiffnessmatrixdict[key] = output
  return output
end


const volumefluxstiffnessmatrixdict = Dict{UInt64, Any}()

function volumefluxstiffnessmatrix(nodesi::NDimNodes, nodesj::NDimNodes, dim::Int)
  N = ndims(nodesi)
  @assert 0 < dim <= ndims(nodesi)
  @assert N == ndims(nodesj)

  key = mapreduce(hash, hash, (nodesi, nodesj, dim))
  haskey(volumefluxstiffnessmatrixdict, key) && return volumefluxstiffnessmatrixdict[key]

  output = ones(length(nodesi), length(nodesj))
  szi = Tuple(length(n) for n in nodesi)
  szj = Tuple(length(n) for n in nodesj)
  output = ones(prod(szi), prod(szj))
  for (j, jj) in enumerate(CartesianIndices(szj)), (i, ii) in enumerate(CartesianIndices(szi))
    tupleii, tuplejj = Tuple(ii), Tuple(jj)
    for d in 1:N
      indi, indj = tupleii[d], tuplejj[d]
      output[i,j] *= _volumefluxstiffnessmatrix(nodesi[d], nodesj[d], indi, indj, d == dim)
    end
  end
  volumefluxstiffnessmatrixdict[key] = output
  return output
end


const _volumefluxstiffnessmatrixdict = Dict{UInt64, Any}()
function _volumefluxstiffnessmatrix(nodesi::AbstractLagrangeNodes, nodesj::AbstractLagrangeNodes, indi, indj, deriv::Bool)

  key = mapreduce(hash, hash, (nodesi, nodesj, indi, indj, deriv))
  haskey(_volumefluxstiffnessmatrixdict, key) && return _volumefluxstiffnessmatrixdict[key]

  output = (if deriv
    QuadGK.quadgk(y->lagrange(y, nodesi, indi) * lagrangederiv(y, nodesj, indj),
                  -1, 1, atol=ATOL, rtol=RTOL)[1]
  else
    QuadGK.quadgk(y->lagrange(y, nodesi, indi) * lagrange(y, nodesj, indj),
                  -1, 1, atol=ATOL, rtol=RTOL)[1]
  end)
  #output *= abs(output > 100ATOL)

  _volumefluxstiffnessmatrixdict[key] = output
  return output
end



function convertbases!(dofsto, dofsfrom, nodesfrom, nodesto)
  @assert size(dofsto) == size(dofsfrom)
  Mtoto = massmatrix(nodesto, nodesto)
  Mtofrom = massmatrix(nodesto, nodesfrom)
  @views dofsto[:] .= Mtoto \ Mtofrom * dofsfrom[:]
end


# derivative of the ind^th nodal lagrange function associated with nodes at position x
function lagrangederiv(x::R, nodes::AbstractLagrangeNodes, ind::Integer) where {R<:Real}
  return sum(i-> 1 / (nodes[ind] - nodes[i]) *
    mapreduce(j->(x - nodes[j]) / (nodes[ind] - nodes[j]), *,
              filter(j->!(j in (ind, i)), eachindex(nodes)); init=one(promote_type(R, eltype(nodes)))),
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


