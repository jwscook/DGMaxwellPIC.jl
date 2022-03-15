
struct Grid{N,T<:BasisFunctionType}
  data::Array{Cell{N, T}, N}
  lower::Vector{Float64}
  upper::Vector{Float64}
  Grid(data::Array{Cell{N, T},N}) where {N,T} = new{N,T}(data,
                                                         collect(minimum(x->x.lower, data)),
                                                         collect(maximum(x->x.upper, data)))
end
Base.getindex(g::Grid, i...) = g.data[i...]
Base.size(g::Grid) = size(g.data)
Base.iterate(g::Grid) = iterate(g.data)
Base.iterate(g::Grid, state) = iterate(g.data, state)
lower(g::Grid) = g.lower
upper(g::Grid) = g.upper
boundingbox(g::Grid) = (lower(g), upper(g))
zero!(g::Grid) = zero!.(g.data)

for (fname, offset, len) ∈ ((:electricfield, 0, 3),
                            (:magneticfield, 3, 3),
                            (:currentfield, 6, 3),
                            (:chargefield, 9, 1))
  fname! = Symbol(fname, :!)
  fnamedofs = Symbol(fname, :dofs)
  fnamedofs! = Symbol(fname, :dofs!)
  @eval function $(fname!)(s::State, data::Union{Tuple, AbstractVector})
    @assert length(data) == $len
    for i in eachindex(data)
      $(fname)(s, i, data[i])
    end
  end
  @eval function $(fnamedofs!)(s::State, data, component::Integer)
    @assert 1 <= component <= $len
    s.q[component + $offset] .= data
  end
  @eval function $(fnamedofs)(s::State, component::Integer)
    @assert 1 <= component <= $len
    return s.q[component + $offset]
  end

  @eval function $(fname)(s::State{N}, x, component) where {N}
    @assert 1 <= component <= $len
    all(-1 <= x[i] <= 1 for i in 1:N) || @warn "x, $x, outside [-1, 1]^n"
    dofs = $(fnamedofs)(s, component)
    sizetuple = size(dofs)
    nodes = gausslegendrenodes(sizetuple)
    output = one(eltype(x))
    return lagrange(x, nodes, dofs)
  end

  @eval function $(fname)(s::State{N}, x) where {N}
    all(-1 <= x[i] <= 1 for i in 1:N) || @warn "x, $x, outside [-1, 1]^n"
    dofs = ndofs(s)
    sizetuple = size(dofs)
    nodes = gausslegendrenodes(sizetuple)
    output = zeros(eltype(x), $(len))
    for component in 1:$(len)
      output[component] = $(fname)(s, x, component)
    end
    return output
  end

  @eval function $(fname!)(s::State{N}, x, component::Integer, value::Real) where {N}
    @assert 1 <= component <= $len
    all(-1 <= x[i] <= 1 for i in 1:N) || @warn "x, $x, outside [-1, 1]^n"
    dofs = $(fnamedofs)(s, component)
    sizetuple = size(dofs)
    nodes = gausslegendrenodes(sizetuple)
    weights = gausslegendreweights(sizetuple)
    output = one(eltype(x))
    lagrange!(dofs, x, nodes, weights, value)
    $(fnamedofs!)(s, dofs, component)
  end

  @eval function $(fname!)(s::State{N}, x, values) where {N}
    @assert length(values) == $(len)
    for i in 1:$(len)
      $(fname!)(s, x, i, values[i])
    end
  end

  @eval function $(fname!)(c::Cell, f::F, component::Integer) where {F<:Function}
    @assert 1 <= component <= $len
    dofs = $(fnamedofs)(state(c), component)
    sizetuple = size(dofs)
    nodes = gausslegendrenodes(sizetuple)
    weights = gausslegendreweights(sizetuple)
    lagrange!(dofs, nodes, weights, f, lower(c), upper(c))
    $(fnamedofs!)(state(c), dofs, component)
  end
  @eval function $(fname!)(c::Cell, f::F) where {F<:Function}
    for component in 1:$len
      $(fname!)(c, x->f(x)[component], component)
    end
  end

  @eval $(fname)(cell::Cell, args...) = $(fname)(state(cell), localx(cell, args[1]), args[2:end]...)
  @eval $(fname!)(cell::Cell, args...) = $(fname!)(state(cell), localx(cell, args[1]), args[2:end]...)

  @eval function $(fname)(g::Grid{N}, args...) where {N}
    x = args[1]
    c = cell(g, x)
    isnothing(c) && return zeros(eltype(x), $(len))
    return $(fname)(state(c), localx(c, x), args[2:end]...)
  end

  @eval function $(fname)(g::Grid, component::Integer)
    output = zeros(size(g))
    for i in CartesianIndices(size(g))
      output[i] = $fname(g[i], centre(g[i]), component)
    end
    return output
  end
  @eval $(fname)(g::Grid) = ($(fname)(g, i) for i in 1:$len)

  @eval function $(fname!)(g::Grid{N}, args...) where {N}
    x = args[1]
    c = cell(g, x)
    isnothing(c) || $(fname!)(state(c), localx(c, x), args[2:end]...)
  end

  @eval function $(fname!)(g::Grid, f::F, component::Integer) where {F<:Function}
    for c in g
      $(fname!)(c, f, component)
    end
  end

  @eval function $(fname!)(g::Grid, f::F) where {F<:Function}
    for component in 1:$len, c in g
      $(fname!)(c, x->f(x)[component], component)
    end
  end

end

function cellcentres(g::Grid{N}) where {N}
  output = Array{Union{Nothing, Any}}(nothing, size(g))
  for i in CartesianIndices(size(g))
    output[collect(Tuple(i))] = centre(g[i])
  end
  return output
end


function cell(g::Grid{N}, x) where {N}
  @assert !any(isnan, x) "x=$x"
  lb = lower(g)
  ub = upper(g)
  sg = size(g)
  index = [Int(ceil(sg[i] * (x[i] - lb[i])/(ub[i] - lb[i]))) for i in eachindex(x)]
  any(index .< 1) && return nothing
  any(index .> size(g)) && return nothing
  c = g[index...]
  @assert in(x, c) "$(lower(c)), $x, $(upper(c))"
  return c
#  for i in CartesianIndices(size(g))
#    c = g[i]
#    in(x, c) && return c
#  end
end

componentindexoffset(cell::Cell, component::Integer) = componentindexoffset(state(cell), component)

function componentindexoffset(s::State, component::Integer)
  1 <= component <= length(s.q) || throw(ArgumentError("Cannot access component $component"))
  ind = 0
  for i in 1:component-1
    ind += length(s.q[component])
  end
  return ind
end

function divergence(g::Grid{N}, x, f::F) where {N, F}
  j = ForwardDiff.jacobian(y->f(g, y), x)
  return sum(j[i, i] for i in 1:N)
end

divB(g::Grid, x) = divergence(g, x, magneticfield)
divE(g::Grid, x) = divergence(g, x, electricfield)

function sources(g::Grid)
  A = volumemassmatrix(g)
  x = currentsource(g)
  return A * x
end


