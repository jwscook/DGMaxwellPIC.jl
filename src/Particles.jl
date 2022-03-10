
abstract type AbstractParticleSampler end

struct Particles{N, T}
  phasespacepositions::Array{T, 2}
  Particles(data::Array{T,2}) where {T} = new{size(data, 2) - 3, T}(data)
end

function Particles(f::F, npart_upperbound, lowerxv, upperxv,
    sampler::AbstractParticleSampler=HaltonSampler()) where {F}
  return Particles(particlephasepositions(f, npart_upperbound, lowerxv, upperxv, sampler))
end

Base.length(p::Particles) = size(p.phasespacepositions, 1)

position(p::Particles{N}, i::Integer) where {N} = p.phasespacepositions[i, 1:N]
velocity(p::Particles{N}, i::Integer) where {N} = p.phasespacepositions[i, N+1:N+3]

for DIM in (3, 2, 1)
  @eval xposition(p::Particles{$DIM}) = @view p.phasespacepositions[:, 1]
  DIM == 3 && continue
  @eval yposition(p::Particles{$DIM}) = @view p.phasespacepositions[:, 2]
end
zposition(p::Particles{N}) where {N} = @view p.phasespacepositions[:, 3]
xvelocity(p::Particles{N}) where {N} = @view p.phasespacepositions[:, N+1]
yvelocity(p::Particles{N}) where {N} = @view p.phasespacepositions[:, N+2]
zvelocity(p::Particles{N}) where {N} = @view p.phasespacepositions[:, N+3]

struct HaltonSampler <: AbstractParticleSampler end
(h::HaltonSampler)(i, p) = haltonvalue(i, p)

struct RandomSampler <: AbstractParticleSampler end
(r::RandomSampler)(i, p) = rand()

sequencesize(h::HaltonSampler, dim) = prod(prime(i) for i in 1:dim)
sequencesize(r::RandomSampler, dim) = 1

function particlephasepositions(distributionfunction::F, npart_upperbound::Integer, lowerxv, upperxv,
    sampler::AbstractParticleSampler=HaltonSampler()) where {F}
  NV = length(lowerxv)
  seqsize = sequencesize(sampler, NV+1)
  particles = zeros(npart_upperbound, NV)
  seqcount = 0
  lastfullsequence = 0
  i = 0
  while i < npart_upperbound
    xv = [sampler(seqcount, prime(i)) for i in 1:NV]  .* (upperxv .- lowerxv) .+ lowerxv
    y = sampler(seqcount, prime(NV+1))
    seqcount += 1
    if y < distributionfunction(xv)
      i += 1
      particles[i, :] .= xv
    end
    if mod(seqcount, seqsize) == 0
      lastfullsequence = i
    end
  end
  particles = particles[1:lastfullsequence, :]
  return particles
end


function haltonparticlephasepositions(f::F, npart_upperbound::Integer, lowerxv, upperxv) where {F}
  return particlephasepositions(f, npart_upperbound, lowerxv, upperxv, HaltonSampler())
end

function randomparticlephasepositions(f::F, nparticles::Integer, lowerxv, upperxv) where {F}
  return particlephasepositions(f, nparticles, lowerxv, upperxv, RandomSampler())
end

