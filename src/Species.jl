struct Species{N, T}
  particledata::ParticleData{N,T}
  charge::Float64
  mass::Float64
  Species(pd::ParticleData{N,T}; charge, mass) where {N,T} = new{N,T}(pd, charge, mass)
end

Base.sort!(s::Species, g::Grid) = sort!(s.particledata, g)

function Base.copyto!(a::Species, b::Species)
  @assert a.charge == b.charge
  @assert a.mass == b.mass
  copyto!(a.particledata, b.particledata)
end

charge(s::Species) = s.charge
mass(s::Species) = s.mass
numberofparticles(s::Species) = length(s.particledata)

function weight!(s::Species, numberdensity, lower, upper)
  physicalnumberofparticles = numberdensity * prod(upper .- lower)
  w = physicalnumberofparticles / numberofparticles(s)
  weight!(s.particledata, w)
end

weight!(s::Species, numberdensity, grid) = weight!(s, numberdensity, lower(grid), upper(grid))

for fname in (:position, :velocity, :weight, :xposition, :yposition, :zposition, :xvelocity, :yvelocity, :zvelocity, :cellids)
  fname! = Symbol(fname, :!)
  @eval $(fname)(s::Species) = $(fname)(s.particledata)
  @eval $(fname)(s::Species, args...) = $(fname)(s.particledata, args...)
  @eval $(fname!)(s::Species) = $(fname!)(s.particledata)
  @eval $(fname!)(s::Species, args...) = $(fname!)(s.particledata, args...)
end
workarrays(s::Species) = (s.particledata.work3, s.particledata.work6a, s.particledata.work6b)
ids(s::Species) = ids(s.particledata)

