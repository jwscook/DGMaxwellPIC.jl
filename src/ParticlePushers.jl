
function borispush!(x, v, qE_m, qB_m, dt)
  v⁻ = @. v + qE_m * dt / 2
  v¹ = v⁻ .+ (cross(v⁻ .+ cross(v⁻, qB_m) * dt / 2, qB_m) ./ (1 + sum(z->z^2, qB_m) * dt^2 / 4) .+ qE_m / 2 ) * dt
  for i in eachindex(x)
    x[i] += (v[i] + v¹[i]) * dt / 2
  end
  @. v = v¹
  return nothing
end

function advect!(x, v, dt)
  for i in eachindex(x)
    x[i] += v[i] * dt
  end
end
