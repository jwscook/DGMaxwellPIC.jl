
struct Grid{N,T<:BasisFunctionType,U}
  data::Array{Cell{N, T,U}, N}
  lower::Vector{Float64}
  upper::Vector{Float64}
  Grid(data::Array{Cell{N, T, U},N}) where {N,T,U} = new{N,T,U}(data,
    collect(minimum(x->x.lower, data)),
    collect(maximum(x->x.upper, data)))
end
Base.getindex(g::Grid, i...) = g.data[i...]
Base.size(g::Grid) = size(g.data)
Base.length(g::Grid) = length(g.data)
Base.eachindex(g::Grid) = eachindex(g.data)
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
  privatefname! = Symbol(:_, fname, :!)
  fnamedofs = Symbol(fname, :dofs)
  fnamedofs! = Symbol(fname, :dofs!)
  incrementfnamedofs! = Symbol(:increment, fname, :dofs!)
  solvefnamedofs! = Symbol(:solve, fname, :dofs!)

  @eval function $(fnamedofs!)(s::State, data, component::Integer)
    @assert 1 <= component <= $len
    s.q[component + $offset] .= data
  end
  @eval function $(incrementfnamedofs!)(s::State, data, component::Integer)
    @assert 1 <= component <= $len
    s.q[component + $offset] .+= data
  end

  @eval function $(fnamedofs)(s::State, component::Integer)
    @assert 1 <= component <= $len
    return s.q[component + $offset]
  end

  @eval function $(fname)(s::State{N, NodeType}, x, component) where {N, NodeType}
    @assert 1 <= component <= $len
    all(-1 <= x[i] <= 1 for i in 1:N) || @warn "x, $x, outside [-1, 1]^n"
    dofs = $(fnamedofs)(s, component)
    sizetuple = size(dofs)
    nodes = NDimNodes(sizetuple, NodeType)
    return lagrange(x, nodes, dofs)
  end

  @eval function $(fname)(s::State{N, NodeType}, x) where {N, NodeType}
    all(-1 <= x[i] <= 1 for i in 1:N) || @warn "x, $x, outside [-1, 1]^n"
    dofs = ndofs(s)
    sizetuple = size(dofs)
    output = zeros(eltype(x), $(len))
    for component in 1:$(len)
      output[component] = $(fname)(s, x, component)
    end
    @assert !any(isnan, output) "output = $output"
    return output
  end

  @eval function $(privatefname!)(dofs, x, component::Integer, value,
      solvedofsornot::MaybeSolveDofs, nodes)
    @assert 1 <= component <= $len
    @assert !any(isnan, x) "x=$x"
    N = length(size(dofs))
    all(-1 <= x[i] <= 1 for i in 1:N) || @warn "x, $x, outside [-1, 1]^n in $(privatefname!)"
    fill!(dofs, 0)
    lagrange!(dofs, x, nodes, value, -(@SArray ones(N)), (@SArray ones(N)), solvedofsornot)
    @assert !any(isnan, dofs) "dofs = $dofs, x = $x, value = $value"
  end

  @eval function $(fname!)(c::Cell{N, NodeType}, x::T, values::U
      ) where {N, NodeType, T<:AbstractArray{<:Number, 1}, U<:AbstractArray{<:Number, 1}}
    @assert length(values) == $(len)
    s = state(c)
    #@warn "This is the slow version"
    dofs = workdofs(s)
    nodes = NDimNodes(size(dofs), NodeType)
    z = referencex(c, x)
    for i in 1:$(len)
      $(privatefname!)(dofs, z, i, values[i], SolveDofsNow(), nodes)
      # below is a is a no-op
      maybesolvedofs!(dofs, nodes, -(@SArray ones(N)), (@SArray ones(N)), DelayDofsSolve())
      $(incrementfnamedofs!)(s, dofs, i)
    end
  end

  @eval function $(fname!)(c::Cell{N, NodeType}, x::T, values::U,
      ) where {N, NodeType, T<:AbstractArray{<:Number, 2}, U<:AbstractArray{<:Number, 2}}
    @assert size(values, 1) == $(len)
    s = state(c)
    dofs = workdofs(s)
    fill!(dofs, 0)
    nodes = NDimNodes(size(dofs), NodeType)
    z = referencex(c, x)
    for i in 1:$(len)
      fill!(dofs, 0)
      $(privatefname!)(dofs, z, i, (@view values[i, :]), DelayDofsSolve(), nodes)
      maybesolvedofs!(dofs, nodes, -(@SArray ones(N)), (@SArray ones(N)), SolveDofsNow())
      dofs .*= jacobian(c)
      $(incrementfnamedofs!)(s, dofs, i)
    end
  end

  @eval function $(fname!)(c::Cell{N, NodeType}, f::F, component::Integer
      ) where {N, NodeType, F<:Function}
    @assert 1 <= component <= $len
    dofs = $(fnamedofs)(state(c), component)
    sizetuple = size(dofs)
    nodes = NDimNodes(sizetuple, NodeType)
    fill!(dofs, 0) # zero out before use
    lagrange!(dofs, nodes, f) * jacobian(c)
    $(fnamedofs!)(state(c), dofs, component) # are dofs a reference, so is this even needed?
  end
  @eval function $(fname!)(c::Cell, f::F) where {F<:Function}
    for component in 1:$len
      $(fname!)(c, x->f(x)[component], component)
    end
  end

  @eval $(fname)(cell::Cell{N,T}, x, args...) where {N,T} = $(fname)(state(cell), referencex(cell, x), args...)
  @eval $(fname!)(cell::Cell{N,T}, x, args...) where {N,T} = $(fname!)(state(cell), referencex(cell, x), args...)

  @eval function $(fname)(g::Grid{N}, args...) where {N}
    x = args[1]
    c = cell(g, x)
    isnothing(c) && return zeros(eltype(x), $(len))
    return $(fname)(state(c), referencex(c, x), args[2:end]...)
  end

  @eval function $(fname)(g::Grid, component::Integer)
    output = zeros(size(g))
    for i in CartesianIndices(size(g))
      output[i] = $(fname)(g[i], centre(g[i]), component)
    end
    return output
  end
  @eval $(fname)(g::Grid) = ($(fname)(g, i) for i in 1:$len)

  @eval function $(fname!)(g::Grid{N}, args...) where {N}
    # this is the slow version that processes each particle at a time and finds its cell
    x = args[1]
    c = cell(g, x)
    isnothing(c) || $(fname!)(state(c), referencex(c, x), args[2:end]...)
  end

  @eval function $(fname!)(g::Grid{N}, cellids::Vector{<:Integer}, x::AbstractArray{<:Number, 2},
       args...) where {N}
    # this is the fast version that processes batches of particle that are sorted into cells
    @assert issorted(cellids) "$cellids"
    isempty(cellids) && return nothing
    i2 = cellids[1] - 1
    for i in eachindex(g)
      i1 = findfirst(==(i), cellids)
      i1 == nothing && continue
      i2 = findlast(==(i), cellids)
      cell = g[i]
      $(fname!)(cell, (@view x[:, i1:i2]), args...) # bugfest args!?!
    end
#    while i2 < length(cellids) # TODO turn this into a for loop and parallelise
#      i1 = i2 + 1
#      cellid = cellids[i1]
#      i2 = findlast(x->isequal(x, cellid), cellids)
#      cell = g[cellid]
#      $(fname!)(cell, (@view x[:, i1:i2]), args...) # bugfest args!?!
#    end
  end

  @eval function $(fname!)(g::Grid, f::F, component::Integer) where {F<:Function}
    for c in g
      $(fname!)(c, x->f(originalx(c, x)), component)
    end
  end

  @eval function $(fname!)(g::Grid, f::F) where {F<:Function}
    for component in 1:$len, c in g
      $(fname!)(c, x->f(originalx(c, x))[component], component)
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

function cellid(g::Grid{N}, x) where {N}
  @assert !any(isnan, x) "x=$x"
  lb = lower(g)
  ub = upper(g)
  sg = size(g)
  index = try
    CartesianIndex((Int(ceil(sg[i] * (x[i] - lb[i])/(ub[i] - lb[i]))) for i in eachindex(x))...)
  catch
    throw(ErrorException("Can't find cell for position $x"))
  end
  any(<(1), Tuple(index)) && return nothing
  any(Tuple(index) .> size(g)) && return nothing
  if !in(x, g[index]) # some crazy floating point inaccuracy?
    for i in CartesianIndices(g.data)
      if in(x, g[i])
        index = i
        break
      end
    end
  end
  return index
end

function cell(g::Grid, x)
  index = cellid(g, x)
  c = g[index]
  @assert in(x, c) "$(lower(c)), $x, $(upper(c))"
  return c
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


