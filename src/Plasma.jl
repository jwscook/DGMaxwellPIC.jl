struct Plasma{T}
  species::Vector{T}
end
Base.eachindex(p::Plasma) = eachindex(p.species)
Base.iterate(p::Plasma) = iterate(p.species)
Base.iterate(p::Plasma, i) = iterate(p.species, i)

function Base.copyto!(a::Plasma, b::Plasma)
  for i in eachindex(a)
    copyto!(a.species[i], b.species[i])
  end
end


function Base.sort!(p::Plasma, g::Grid)
  for s in p
    sort!(s, g)
  end
  return p
end

function depositcurrent!(g::Grid{N}, plasma::Plasma) where N
  currentfielddofs!(g, 0) # zero current
  #@assert issorted(plasma)
  for species in plasma
    v = velocity(species)
    w = weight(species)
    q = charge(species)
    j, _, _ = workarrays(species)
    @threads for jj in 1:size(v, 2)
       @inbounds for i in 1:N
        j[i, jj] = q * w[jj] * v[i, jj]
      end
    end
    cids = cellids(species)
    x = position(species)
    currentfield!(g, cids, x, j) # add all current values cell-by-cell
  end
end

function periodicbcsapply!(x, g::Grid)
  lb = lower(g)
  ub = upper(g)
  @. x = mod(x - lb, ub - lb) + lb
  for i in eachindex(x)
    x[i] == lb[i] && (x[i] = ub[i])
  end
end

function advance!(plasma::Plasma, g::Grid{N}, dt, gg=missing, tfrac=nothing) where {N}
  lb = lower(g)
  ub = upper(g)
  for species in plasma
    q_m = charge(species) / mass(species)
    x = position(species)
    v = velocity(species)
    _, EB, EB2 = workarrays(species)
    @inbounds @views @threads for i in 1:numberofparticles(species)
      EB[:, i] .= zero(eltype(EB))
      ismissing(gg) || (EB2[:, i] .= zero(eltype(EB2)))
      advect!(x[:, i], v[:, i], dt/2)
      periodicbcsapply!(x[:, i], g)
    end
    sort!(species, g) # sort into cells to get EB field efficiently
    electromagneticfield!(EB, g, cellids(species), x)
    ismissing(gg) || electromagneticfield!(EB2, gg, cellids(species), x)
    @inbounds @views @threads for i in 1:numberofparticles(species)
      E = SVector{3, Float64}(EB[1:3, i])
      B = SVector{3, Float64}(EB[4:6, i])
      ismissing(gg) || (E = E * (1-tfrac) + SVector{3, Float64}(EB2[1:3, i]) * tfrac)
      ismissing(gg) || (B = B * (1-tfrac) + SVector{3, Float64}(EB2[4:6, i]) * tfrac)
      qE_m = E * q_m
      qB_m = B * q_m
      borisrotate!(v[:, i], qE_m, qB_m, dt, Val(N))
      advect!(x[:, i], v[:, i], dt/2)
      periodicbcsapply!(x[:, i], g)
    end
    sort!(species, g) # sort again after move
  end
  return plasma
end

function borisrotate!(v::AbstractVector{T}, qE_m, qB_m, dt, ::Val{N}
    ) where {T, N}
  dt_2 = dt / 2
  v⁻ = SVector{3, T}(v) + qE_m * dt_2
  t = qB_m * dt_2
  s = 2t / (1 + dot(t, t))
  v⁺ = v⁻ + cross(v⁻ + cross(v⁻, t), s)
  v .= v⁺ + qE_m * dt_2
  return nothing
end

function advect!(x, v, dt)
  @inbounds for i in eachindex(x)
    x[i] += v[i] * dt
  end
end
