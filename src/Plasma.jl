struct Plasma
  species::Vector
end
Base.iterate(p::Plasma) = iterate(p.species)
Base.iterate(p::Plasma, i) = iterate(p.species, i)

function depositcurrent!(g::Grid, plasma::Plasma)
  for species in plasma
    x = position(species)
    j = weightedcurrent(species)
    for i in 1:numberofparticles(species)
      currentfield!(g, (@view x[i, :]), (@view j[i, :]))
    end
  end
end

function push!(plasma::Plasma, g::Grid, dt)
  for species in plasma
    q_m = charge(species) / mass(species)
    x = position(species)
    j = weightedcurrent(species)
    for i in 1:numberofparticles(species)
      advect!((@view x[i, :]), (@view v[i, :]), dt/2)
      qE_m = electricfield(g, (@view x[i, :])) * q_m
      qB_B = magneticfield(g, (@view x[i, :])) * q_m
      pushboris!(@view x[i,:], (@view v[i, :]), qE_m, qB_m, dt)
      advect!((@view x[i, :]), (@view v[i, :]), dt/2)
    end
  end
end


