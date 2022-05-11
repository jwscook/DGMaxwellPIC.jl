
struct Grid{N,T<:BasisFunctionType,U}
  data::Array{Cell{N, T,U}, N}
  lower::Vector{Float64}
  upper::Vector{Float64}
  dofoffsetdict::Dict{NTuple{N,Int}, Int}
  dofrangedict::Dict{NTuple{N,Int}, UnitRange{Int}}
  ndimnodesdict::Dict{NTuple{N,Int}, NDimNodes{N,T}}
  lummdict::Dict{UInt64, LU{Float64, Matrix{Float64}}} # lu of mass matrices
  function Grid(data::Array{Cell{N, T, U},N}) where {N,T,U}
    return new{N,T,U}(data,
    collect(minimum(x->x.lower, data)),
    collect(maximum(x->x.upper, data)),
    lookupdicts(data, T)...)
  end
end
Base.getindex(g::Grid, i...) = g.data[i...]
Base.size(g::Grid) = size(g.data)
Base.size(g::Grid, i) = size(g.data, i)
Base.length(g::Grid) = length(g.data)
numelements(g::Grid) = length(g.data)
Base.eachindex(g::Grid) = eachindex(g.data)
Base.iterate(g::Grid) = iterate(g.data)
Base.iterate(g::Grid, state) = iterate(g.data, state)
lower(g::Grid) = g.lower
upper(g::Grid) = g.upper
boundingbox(g::Grid) = (lower(g), upper(g))
volume(g::Grid) = prod(upper(g) .- lower(g))
zero!(g::Grid) = zero!.(g.data)
indices(grid::Grid, cellindex) = grid.dofrangedict[Tuple(cellindex)]
offsetindex(grid, cellindex) = grid.dofoffsetdict[Tuple(cellindex)]
function ndimnodes(grid::Grid{N,T}, c::Cell{N,T}) where {N,T<:BasisFunctionType}
  return grid.ndimnodesdict[dofshape(c)]
end
ndimnodes(grid::Grid, i) = ndimnodes(grid, grid[i])

function lumassmatrixkey(cell::Cell{N, T}, ignore::Int) where {N, T}
  return mapreduce(hash, hash, (jacobian(cell; ignore=ignore), dofshape(cell), T))
end
function lumassmatrix!(g::Grid{N,T}, cell::Cell{N, T}, ignore::Int=0) where {N, T}
  key = lumassmatrixkey(cell, ignore)
  if !haskey(g.lummdict, key)
    throw(ErrorException("This needs to be made threadsafe"))
    #g.lummdict[key] = lu(kron(I(6), massmatrix(ndimnodes(g, cell)) * jacobian(cell; ignore=ignore)))
  end
  return g.lummdict[key]
end



function lookupdicts(griddata::AbstractArray{T,N}, ::Type{NodeType}) where {T, N, NodeType}
  dofoffsets = Dict{NTuple{N,Int}, Int}()
  dofranges = Dict{NTuple{N,Int}, UnitRange{Int}}()
  ndimnodes = Dict{NTuple{N,Int}, NDimNodes{N,NodeType}}()
  lumms = Dict{UInt64, LU{Float64, Matrix{Float64}}}()
  count = 0
  lastndofscount = 0
  for cellindex in CartesianIndices(griddata)
    count += 1
    ndofscount = lastndofscount + ndofs(griddata[cellindex])
    key = Tuple(cellindex)
    dofoffsets[key] = lastndofscount
    dofranges[key] = (1:ndofs(griddata[cellindex])) .+ dofoffsets[key]
    lastndofscount = ndofscount
    cell = griddata[cellindex]
    sizetuple = dofshape(cell)
    haskey(ndimnodes, sizetuple) || (ndimnodes[sizetuple] = NDimNodes(sizetuple, NodeType))
    for ignore in 0:N
      lkey = lumassmatrixkey(cell, ignore)
      if !haskey(lumms, lkey)
        lumms[lkey] = lu(kron(I(6), massmatrix(ndimnodes[sizetuple]) * jacobian(cell; ignore=ignore)))
      end
    end
  end
  return dofoffsets, dofranges, ndimnodes, lumms
end


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
  @eval function $(fnamedofs!)(s::State, data)
    foreach(component->$(fnamedofs!)(s, data, component), 1:$len)
  end

  @eval function $(incrementfnamedofs!)(s::State, data, component::Integer)
    @assert 1 <= component <= $len
    s.q[component + $offset] .+= data
  end

  @eval function $(fnamedofs)(s::State, component::Integer)
    @assert 1 <= component <= $len
    return s.q[component + $offset]
  end

  @eval function $(fname)(s::State{N, NodeType}, nodes, x, component) where {N, NodeType}
    @assert 1 <= component <= $len
    all(-1 <= x[i] <= 1 for i in 1:N) || @warn "x, $x, outside [-1, 1]^n"
    dofs = $(fnamedofs)(s, component)
    sizetuple = size(dofs)
    return lagrange(x, nodes, dofs)
  end

  @eval function $(fname!)(output::AbstractVector, s::State{N, NodeType}, nodes, x) where {N, NodeType}
    all(-1 <= x[i] <= 1 for i in 1:N) || @warn "x, $x, outside [-1, 1]^n"
    dofs = ndofs(s)
    sizetuple = size(dofs)
    for component in 1:$(len)
      output[component] = $(fname)(s, nodes, x, component)
    end
    @assert !any(isnan, output) "output = $output"
    return output
  end

  @eval function $(fname)(s::State{N, NodeType}, nodes, x) where {N, NodeType}
    output = @MArray zeros(eltype(x), $(len))
    fname!(output, s, nodes, x)
    output
  end

  @eval function $(fname!)(output::AbstractMatrix, s::State{N, NodeType}, nodes, x) where {N, NodeType}
    @assert size(output, 2) == size(x, 2) "$(size(output, 2)), $(size(x, 2))"
    @views for i in axes(output, 2)
      $(fname!)(output[:, i], s, nodes, x[:, i])
    end
    return output
  end

  @eval function $(privatefname!)(dofs, x, component::Integer, value,
      solvedofsornot::MaybeSolveDofs, nodes)
    @assert 1 <= component <= $len
    @assert !any(isnan, x) "x=$x"
    N = length(size(dofs))
    all(-1 <= x[i] <= 1 for i in 1:N) || @warn "x, $x, outside [-1, 1]^n in $(privatefname!)"
    fill!(dofs, 0)
    lagrange!(dofs, x, nodes, value, solvedofsornot)
    @assert !any(isnan, dofs) "dofs = $dofs, x = $x, value = $value"
  end

#  @eval function $(fname!)(c::Cell{N, NodeType}, nodes, x::T, values::U
#      ) where {N, NodeType, T<:AbstractArray{<:Number, 1}, U<:AbstractArray{<:Number, 1}}
#    @assert length(values) == $(len)
#    @warn "This is the slow version"
#    s = state(c)
#    dofs = workdofs(s)
#    z = referencex(x, c)
#    for i in 1:$(len)
#      $(privatefname!)(dofs, z, i, values[i], SolveDofsNow(), nodes)
#      # below is a is a no-op
#      maybesolvedofs!(dofs, nodes, DelayDofsSolve())
#      $(incrementfnamedofs!)(s, dofs, i)
#    end
#  end

  @eval function $(fname!)(c::Cell{N, NodeType}, nodes, x::T, values::U,
      ) where {N, NodeType, T<:AbstractArray{<:Number, 2}, U<:AbstractArray{<:Number, 2}}
    @assert size(values, 1) == $(len)
    s = state(c)
    dofs = workdofs(s)
    #z = referencex(x, c)
    referencex!(x, c)
    for i in 1:$(len)
      fill!(dofs, 0)
      #$(privatefname!)(dofs, z, i, (@view values[i, :]), DelayDofsSolve(), nodes)
      $(privatefname!)(dofs, x, i, (@view values[i, :]), DelayDofsSolve(), nodes)
      maybesolvedofs!(dofs, nodes, SolveDofsNow())
      dofs .*= jacobian(c)
      $(incrementfnamedofs!)(s, dofs, i)
    end
    originalx!(x, c)
  end

  @eval function $(fname!)(c::Cell{N, NodeType}, nodes, f::F, component::Integer
      ) where {N, NodeType, F<:Function}
    @assert 1 <= component <= $len
    dofs = $(fnamedofs)(state(c), component)
    sizetuple = size(dofs)
    fill!(dofs, 0) # zero out before use
    lagrange!(dofs, nodes, x->f(originalx(x, c)))
    $(fnamedofs!)(state(c), dofs, component) # are dofs a reference, so is this even needed?
  end
  @eval function $(fname!)(c::Cell, nodes, f::F) where {F<:Function}
    for component in 1:$len
      $(fname!)(c, nodes, x->f(originalx(x, c))[component], component)
    end
  end

  @eval $(fname)(cell::Cell{N,T}, n, x, args...) where {N,T} = $(fname)(state(cell), n, referencex(x, cell), args...)
  @eval $(fname!)(cell::Cell{N,T}, n, x, args...) where {N,T} = $(fname!)(state(cell), n, referencex(x, cell), args...)
  @eval $(fname!)(output, cell::Cell{N,T}, n, x, args...) where {N,T} = $(fname!)(output, state(cell), n, referencex(x, cell), args...)

  @eval function $(fname)(g::Grid{N}, args...) where {N}
    x = args[1]
    c = cell(g, x)
    nodes = ndimnodes(g, c)
    isnothing(c) && return zeros(eltype(x), $(len))
    return $(fname)(state(c), nodes, referencex(x, c), args[2:end]...)
  end

  @eval function $(fname)(g::Grid, component::Integer)
    output = zeros(size(g))
    for i in CartesianIndices(size(g))
      nodes = ndimnodes(g, i)
      output[i] = $(fname)(g[i], nodes, centre(g[i]), component)
    end
    return output
  end
  @eval $(fname)(g::Grid) = ($(fname)(g, i) for i in 1:$len)

  @eval function $(fname!)(g::Grid{N}, args...) where {N}
    # this is the slow version that processes each particle at a time and finds its cell
    x = args[1]
    c = cell(g, x)
    isnothing(c) || $(fname!)(state(c), ndimnodes(g, c), referencex(x, c), args[2:end]...)
  end

  @eval function $(fnamedofs!)(g::Grid{N}, data, component) where {N}
    foreach(i->$(fnamedofs!)(state(g[i]), data, component), eachindex(g))
  end
  @eval function $(fnamedofs!)(g::Grid{N}, data) where {N}
    foreach(i->foreach(component->$(fnamedofs!)(state(g[i]), data, component), 1:$len), eachindex(g))
  end

  @eval function $(fname!)(g::Grid{N}, cellids::Vector{<:Integer}, x::AbstractArray{<:Number, 2},
       args...) where {N}
    # this is the fast version that processes batches of particle that are sorted into cells
    @assert issorted(cellids) "$cellids"
    isempty(cellids) && return nothing
    @views @threads for i in eachindex(g)
      i1 = findfirst(==(i), cellids)
      i1 == nothing && continue
      i2 = findlast(==(i), cellids)
      cell = g[i]
      $(fname!)(cell, ndimnodes(g, cell), x[:, i1:i2], args...) # bugfest args!?!
    end
  end

  @eval function $(fname!)(output::AbstractArray, g::Grid{N}, cellids::Vector{<:Integer},
      x::AbstractArray{<:Number, 2}, args...) where {N}
    # this is the fast version that processes batches of particle that are sorted into cells
    @assert issorted(cellids) "$cellids"
    isempty(cellids) && return nothing
    @views @threads for i in eachindex(g)
      i1 = findfirst(==(i), cellids)
      i1 == nothing && continue
      i2 = findlast(==(i), cellids)
      cell = g[i]
      $(fname!)(output[:, i1:i2], cell, ndimnodes(g, cell), x[:, i1:i2], args...) # bugfest args!?!
    end
  end


  @eval function $(fname!)(g::Grid, f::F, component::Integer) where {F<:Function}
    for c in g
      $(fname!)(c, ndimnodes(g, c), f, component)
    end
  end

  @eval function $(fname!)(g::Grid, f::F) where {F<:Function}
    for component in 1:$len, c in g
      $(fname!)(c, ndimnodes(g, c), x->f(x)[component], component)
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
  lb = lower(g)
  ub = upper(g)
  sg = size(g)
  index = NTuple{N, Int}(Int(ceil(sg[i] * (x[i] - lb[i])/(ub[i] - lb[i]))) for i in eachindex(x))
  in(x, g[index...]) && return index
  if ! # floating point inaccuracy has happened so work out cell by brute force
    for i in CartesianIndices(g.data)
      if in(x, g[i])
        return Tuple{N, Int}(i)
      end
    end
  end
  throw(ErrorException("Shouldnt be able to get here: $lb, $x, $ub"))
end

function cell(g::Grid, x)
  index = cellid(g, x)
  c = g[index...]
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

sources(g::Grid) = -speedoflight^2 * mu0 * currentsource(g)


