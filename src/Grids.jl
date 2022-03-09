
struct Grid{N,T<:BasisFunctionType}
  data::Array{Cell{N, T}, N}
end
Base.getindex(g::Grid, i...) = g.data[i...]
Base.size(g::Grid) = size(g.data)
Base.iterate(g::Grid) = iterate(g.data)
Base.iterate(g::Grid, state) = iterate(g.data, state)
boundingbox(g::Grid) = (minimum(x->x.min, g), maximum(x->x.max, g))
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
  @eval function $(fnamedofs!)(s::State, component::Integer, data)
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
    output = one(eltype(x))
    lagrange!(dofs, x, nodes, value)
    $(fnamedofs!)(s, component, dofs)
  end

  @eval function $(fname!)(s::State{N}, x, values) where {N}
    @assert length(values) == $(len)
    for i in 1:$(len)
      $(fname!)(s, x, i, values[i])
    end
  end

  @eval $(fname)(cell::Cell{N}, args...) where {N} = $(fname)(state(cell), args...)
  @eval $(fname!)(cell::Cell{N}, args...) where {N} = $(fname!)(state(cell), args...)

  @eval function $(fname)(g::Grid{N}, args...) where {N}
    x = args[1]
    c = cell(g, x)
    isnothing(c) && return zeros(eltype(x), $(len))
    return $(fname)(state(c), localx(c, x), args[2:end]...)
  end

  @eval function $(fname!)(g::Grid{N}, args...) where {N}
    x = args[1]
    c = cell(g, x)
    isnothing(c) || $(fname!)(state(c), localx(c, x), args[2:end]...)
  end

end

function cell(g::Grid, x)
  for c in g
    in(x, c) && return c
  end
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

