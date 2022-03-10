struct Species{N, T, W}
  particledata::ParticleData{N,T}
  charge::Float64
  mass::Float64
end

numberofparticles(s::Species) = length(s.particledata)

function weight!(s::Species, numberdensity, lower, upper)
  physicalnumberofparticles = numberdensity * prod(upper .- lower)
  w = physicalnumberofparticles / numberofparticles(s)
  weight!(s.particledata, w)
end

weight!(s::Species, numberdensity, grid) = weight!(s, numberdensity, lower(grid), upper(grid))

