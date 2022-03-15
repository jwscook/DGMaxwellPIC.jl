struct Plasma
  species::Vector
end
Base.iterate(p::Plasma) = iterate(p.species)
Base.iterate(p::Plasma, i) = iterate(p.species, i)

function depositcurrent!(g::Grid, plasma::Plasma)
  for species in plasma
    x = position(species)
    v = velocity(species)
    w = weight(species)
    q = charge(species)
    #j = (@~ q * (w' .* v)) # create a lazy array
    #j = q * (w' .* v) # create a lazy array
    tmp = zeros(size(v, 1))
    for i in 1:numberofparticles(species)
      #currentfield!(g, (@view x[:, i]), (@view j[:, i]))
      currentfield!(g, (@view x[:, i]), broadcast!(*, tmp, (@view v[:, i]), w[i], q))
    end
  end
end

function advance!(plasma::Plasma, g::Grid, dt)
  lb = lower(g)
  ub = upper(g)
  for species in plasma
    q_m = charge(species) / mass(species)
    x = position(species)
    v = velocity(species)
    for i in 1:numberofparticles(species)
      advect!((@view x[:, i]), (@view v[:, i]), dt/2)
      @views x[:,i] .= mod.(x[:,i] .- lb, ub - lb) + lb
      qE_m = electricfield(g, (@view x[:, i])) * q_m
      qB_m = magneticfield(g, (@view x[:, i])) * q_m
      borispush!((@view x[:, i]), (@view v[:, i]), qE_m, qB_m, dt)
      advect!((@view x[:, i]), (@view v[:, i]), dt/2)
      @views x[:,i] .= mod.(x[:,i] .- lb, ub - lb) + lb
    end
  end
end


