
function borispush!(x, v, qE_m, qB_m, dt)
  v⁻ = @. v + qE_m * dt / 2
  v¹ = v⁻ .+ (cross(v⁻ .+ cross(v⁻, qB_m) * dt / 2, qB_m) ./ (1 + sum(z->z^2, qB_m) * dt^2 / 4) .+ qE_m / 2 ) * dt
  @. x += (v + v¹) * dt / 2
  @. v = v¹
  return nothing
end

