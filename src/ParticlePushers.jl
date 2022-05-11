

function borispush!(x::AbstractVector{T}, v::AbstractVector{T}, qE_m, qB_m, dt, ::Val{N}) where {T, N}
  dt_2 = dt / 2
  v⁻ = SVector{3, T}(v) + SVector{3, T}(qE_m) * dt_2
  v⁻×qB_m = cross(v⁻, SVector{3, T}(qB_m))
  denom = (1 + sum(abs2, qB_m) * dt_2^2)
  v .= v⁻ .+ (cross(v⁻ .+ v⁻×qB_m * dt_2, qB_m) ./ denom .+ qE_m ./ 2) .* dt
  return nothing
end

function advect!(x, v, dt)
  @inbounds for i in eachindex(x)
    x[i] += v[i] * dt
  end
end
