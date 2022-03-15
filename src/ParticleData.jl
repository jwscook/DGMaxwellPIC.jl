
abstract type AbstractParticleSampler end

struct ParticleData{N, T<:AbstractArray}
  data::T
  ParticleData(data::Array{T,2}) where {T} = new{size(data, 1) - 4, typeof(data)}(data)
end

function ParticleData(f::F, npart_upperbound, lowerxv, upperxv,
    sampler::AbstractParticleSampler=HaltonSampler()) where {F}
  return ParticleData(particledata(f, npart_upperbound, lowerxv, upperxv, sampler))
end

Base.length(p::ParticleData) = size(p.data, 2)

position(p::ParticleData{N}, i::Integer) where {N} = p.data[1:N, i]
velocity(p::ParticleData{N}, i::Integer) where {N} = p.data[N+1:N+3, i]
weight(p::ParticleData{N}, i::Integer) where {N} = p.data[N+4, i]

position!(p::ParticleData{N}, x, i::Integer) where {N} = (p.data[1:N, i] .= x; p)
velocity!(p::ParticleData{N}, v, i::Integer) where {N} = (p.data[N+1:N+3, i] .= v; p)
weight!(p::ParticleData{N}, w, i::Integer) where {N} = (p.data[N+4, ] .= w; p)

position(p::ParticleData{N}) where {N} = @view p.data[1:N, :]
velocity(p::ParticleData{N}) where {N} = @view p.data[N+1:N+3, :]
weight(p::ParticleData{N}) where {N} = @view p.data[N+4, :]

for DIM in (3, 2, 1)
  @eval xposition(p::ParticleData{$DIM}) = @view p.data[1, :]
  @eval xposition!(p::ParticleData{$DIM}, x) = (@views p.data[1, :] .= x; p)
  DIM == 3 && continue
  @eval yposition(p::ParticleData{$DIM}) = @view p.data[2, :]
  @eval yposition!(p::ParticleData{$DIM}, y) = (@views p.data[2, :] .= y; p)
end
zposition(p::ParticleData{N}) where {N} = @view p.data[3, :]
zposition!(p::ParticleData{N}, z) where {N} = (@views p.data[3, :] .= z; p)

xvelocity(p::ParticleData{N}) where {N} = @view p.data[N+1, :]
yvelocity(p::ParticleData{N}) where {N} = @view p.data[N+2, :]
zvelocity(p::ParticleData{N}) where {N} = @view p.data[N+3, :]

xvelocity!(p::ParticleData{N}, v) where {N} = (@views p.data[N+1, :] .= v; p)
yvelocity!(p::ParticleData{N}, v) where {N} = (@views p.data[N+2, :] .= v; p)
zvelocity!(p::ParticleData{N}, v) where {N} = (@views p.data[N+3, :] .= v; p)
weight!(p::ParticleData{N}, w) where {N} = (@views p.data[N+4, :] .= w; p)


struct HaltonSampler <: AbstractParticleSampler end
(h::HaltonSampler)(i, p) = haltonvalue(i, p)

struct RandomSampler <: AbstractParticleSampler end
(r::RandomSampler)(i, p) = rand()

sequencesize(h::HaltonSampler, dim) = prod(prime(i) for i in 1:dim)
sequencesize(r::RandomSampler, dim) = 1

function particledata(distributionfunction::F, npart_upperbound::Integer, lowerxv, upperxv,
    sampler::AbstractParticleSampler=HaltonSampler(), offsets=rand(length(lowerxv))) where {F}
  NXV = length(lowerxv)
  NXVW = NXV + 1
  seqsize = sequencesize(sampler, NXV+1)
  particles = zeros(NXVW, npart_upperbound)
  seqcount = 0
  lastfullsequence = 0
  i = 0
  while i < npart_upperbound
    xv = mod.([sampler(seqcount, prime(i)) for i in 1:NXV] .+ offsets, 1)
    @assert all(0 .<= xv[:] .<= 1)
    xv .= xv .* (upperxv .- lowerxv) .+ lowerxv
    y = sampler(seqcount, prime(NXV+1))
    seqcount += 1
    if y < distributionfunction(xv)
      i += 1
      particles[1:NXV, i] .= xv
    end
    if mod(seqcount, seqsize) == 0
      lastfullsequence = i
    end
  end
  particles = particles[:, 1:lastfullsequence]
  return particles
end


function haltonparticledata(f::F, npart_upperbound::Integer, lowerxv, upperxv) where {F}
  return particledata(f, npart_upperbound, lowerxv, upperxv, HaltonSampler())
end

function randomparticledata(f::F, nparticles::Integer, lowerxv, upperxv) where {F}
  return particledata(f, nparticles, lowerxv, upperxv, RandomSampler())
end

