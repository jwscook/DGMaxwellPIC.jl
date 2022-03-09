
function lagrange(x::T, nodes, ind::Integer) where {T<:Real}
  otherindices = filter(j->!(j in ind), eachindex(nodes))
  return prod(x .- nodes[otherindices]) / prod(nodes[ind] .- nodes[otherindices])
end

function lagrangederiv(x::T, nodes, ind::Integer) where {T<:Real}
  return sum(i-> 1 / (nodes[ind] .- nodes[i]) *
    mapreduce(j->(x .- nodes[j]) / (nodes[ind] .- nodes[j]), *,
              filter(j->!(j in (ind, i)), eachindex(nodes)); init=one(T)),
    filter(i->!(i in ind), eachindex(nodes)))
end


for (fname) ∈ (:lagrange, :lagrangederiv)
  @eval $(fname)(x::Real, nodes, ind::Integer, coeff::Number) = $(fname)(x, nodes, ind) * coeff
end

function lagrange(x, nodes, inds, coeff::Number)
  return prod(lagrange(x[i], nodes[i], inds[i]) for i in 1:length(x)) * coeff
end

function lagrange(x, nodes, coeffs)
  output = zero(promote_type(eltype(x), eltype(coeffs)))
  for i in CartesianIndices(coeffs)
    t = Tuple(i)
    output += lagrange(x, nodes, Tuple(i), coeffs[i])
  end
  return output
end

function lagrangederiv(x, nodes, inds, coeff::Number, derivdirection)
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


function lagrange!(coeffs, x, nodes, value)
  for i in CartesianIndices(coeffs)
    t = Tuple(i)
    coeffs[i] += lagrange(x, nodes, Tuple(i), value)
  end
end

