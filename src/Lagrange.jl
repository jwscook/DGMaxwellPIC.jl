
const AVAV = AbstractVector{<:AbstractVector}
const AVN = AbstractVector{<:Number}

function gausslegendrenodes(s, a=-ones(length(s)), b=ones(length(s)))
  return [(gausslegendre(si)[1] .+ 1) ./2 .* (b[i] - a[i]) .+ a[i]  for (i, si) in enumerate(s)]
end
function gausslegendreweights(s, a=-ones(length(s)), b=ones(length(s)))
  return [gausslegendre(si)[2] .* (b[i] - a[i])/2  for (i, si) in enumerate(s)]
end

#evaluate one function without coefficient
function lagrange(x::R, nodes::AVN, ind::Integer) where {R<:Real}
  T = promote_type(R, eltype(nodes))
  output = one(T)
  for j in eachindex(nodes)
    j == ind && continue
    output *= (x - nodes[j]) / (nodes[ind] - nodes[j])
  end
  return output
  #otherindices = filter(j->!(j in ind), eachindex(nodes))
  #return prod(x .- nodes[otherindices]) / prod(nodes[ind] .- nodes[otherindices])
end


#evaluate one function with coefficient
for (fname) ∈ (:lagrange, :lagrangederiv)
  @eval $(fname)(x::Real, nodes::AVN, ind::Integer, coeff::Number) = $(fname)(x, nodes, ind) * coeff
end

#evaluate one cartesian product of functions for one coefficient
function lagrange(x, nodes::AVAV, inds, coeff::Number)
  return prod(lagrange(x[i], nodes[i], inds[i]) for i in 1:length(x)) * coeff
end

# evaluate all functions with all coefficients
function lagrange(x, nodes::AVAV, coeffs)
  output = zero(promote_type(eltype(x), eltype(coeffs)))
  for i in CartesianIndices(coeffs)
    output += lagrange(x, nodes, Tuple(i), coeffs[i])
  end
  return output
end

referencex(x, a, b) = @. (x - a) / (b - a) * 2 - 1
globalx(x, a, b) = @. (x / 2 + 1) * (b - a) + a


# incrememnt lagrange coefficients by value as if a function value * DirecDelta(x)
# hence division by the weight, w.
function lagrange!(coeffs, x, nodes::AVAV, weights::AVAV, value)
  for i in CartesianIndices(coeffs)
    t = Tuple(i)
    w = prod(weights[i][t[i]] for i in 1:length(x))
    coeffs[i] += lagrange(x, nodes, Tuple(i), value) / w
  end
end
function lagrange!(coeffs, nodes::AVAV, weights::AVAV, f::F, a, b) where {F<:Function}
  physicalarea = prod(b .- a)
  referencearea = 2^length(a)
  arearatio = referencearea / physicalarea
  @floop for i in CartesianIndices(coeffs)
    t = Tuple(i)
    w = prod(weights[j][t[j]] for j in 1:length(t))
    @assert w > 0
    integral = HCubature.hcubature(y->f(y) * lagrange(referencex(y, a, b), nodes, t, 1), a, b)[1]
    coeffs[i] = integral / w * arearatio
  end
end


# derivative of the ind^th nodal lagrange function associated with nodes at position x
function lagrangederiv(x::Real, nodes::AVN, ind::Integer)
  return sum(i-> 1 / (nodes[ind] - nodes[i]) *
    mapreduce(j->(x - nodes[j]) / (nodes[ind] - nodes[j]), *,
              filter(j->!(j in (ind, i)), eachindex(nodes)); init=one(T)),
    filter(i->!(i in ind), eachindex(nodes)))
end


# evaluate derivative along the direction, dirdiection, of the nodal lagrange function
# product associated with `coeff` for `nodes` at position x
function lagrangederiv(x::AVAV, nodes::AVAV, inds::AVN, coeff::Number, derivdirection::Integer)
  output = one(promote_type(eltype(x), typeof(coeff)))
  for i in 1:length(x)
    if derivdirection == i
      output *= (lagrangederiv(x[i], nodes[i], inds[i]) for i in 1:length(x)) * coeff
    else
      output *= (lagrange(x[i], nodes[i], inds[i]) for i in 1:length(x)) * coeff
    end
  end
  return output
end

function lagrangederiv(x::AVAV, nodes::AVAV, coeffs::AbstractArray, derivdirection::Integer)
  return mapreduce(i->lagrangederiv(x, nodes, Tuple(i), coeffs[i], derivdirection), +, CartesianIndices(coeffs))
end


