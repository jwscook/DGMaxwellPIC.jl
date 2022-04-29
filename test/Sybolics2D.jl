using DGMaxwellPIC, Plots, TimerOutputs, StaticArrays, LinearAlgebra

const NX = 3;
const NY = 3;

const OX = 2;
const OY = 2;

const state2D = State([OX, OY], LobattoNodes);

const DIMS = 2
const L = NX * 2;

const a = zeros(DIMS);
const b = ones(DIMS) .* L;

gridposition(x) = ((x .* (b .- a) .+ a))

const g = Grid([Cell(deepcopy(state2D), gridposition(((i-1)/NX, (j-1)/NY)), gridposition((i/NX, j/NY))) for i in 1:NX, j in 1:NY]);


const dtc = minimum((b .- a)./NX./OX) / DGMaxwellPIC.speedoflight
const dt = dtc * 0.2
const upwind = 0

# du/dt = A * u
# u1 - u0 = dt * (A * u)
# u1 - u0 = dt * (A * (u1 + u0)/2)
# u1 = u0 + dt/2 * A * u1 + dt/2 * A * u0
# (1 - dt/2 * A)*u1 = (1 + dt/2 * A) * u0
# u1 = (1 - dt/2 * A)^-1 (1 + dt/2 * A) * u0

const M = assemble(g, upwind=upwind)
const C = M * dt / 2;
const B = lu(I - C);
const A_CN = B \ Matrix(I + C);
const A_explicit = I + assemble(g, upwind=upwind) * dt;

vmm = DGMaxwellPIC.volumemassmatrix(g)
vfsm = DGMaxwellPIC.volumefluxstiffnessmatrix(g)
sfsm = DGMaxwellPIC.surfacefluxstiffnessmatrix(g)

using Plots
heatmap(Matrix(vmm), yflip=true)
xticks!(1:6*OX*NX)
yticks!(1:6*OX*NX)
savefig("vmm.pdf")

heatmap(log10.(abs.(Matrix(vmm))), yflip=true)
xticks!(1:6*OX*NX)
yticks!(1:6*OX*NX)
savefig("vmmlog10abs.pdf")

heatmap(Matrix(vfsm), yflip=true)
xticks!(1:6*OX*NX)
yticks!(1:6*OX*NX)
savefig("vfsm.pdf")

heatmap(Matrix(sfsm), yflip=true)
xticks!(1:6*OX*NX)
yticks!(1:6*OX*NX)
savefig("sfsm.pdf")

heatmap(log10.(abs.(Matrix(M))), yflip=true)
xticks!(1:6*OX*NX)
yticks!(1:6*OX*NX)
savefig("m.pdf")

using Symbolics
Ex11 = map(j -> (i = Tuple(j); @variables Ex11($i))[1], CartesianIndices((OX, OY)))
Ey11 = map(j -> (i = Tuple(j); @variables Ey11($i))[1], CartesianIndices((OX, OY)))
Ez11 = map(j -> (i = Tuple(j); @variables Ez11($i))[1], CartesianIndices((OX, OY)))
Bx11 = map(j -> (i = Tuple(j); @variables Bx11($i))[1], CartesianIndices((OX, OY)))
By11 = map(j -> (i = Tuple(j); @variables By11($i))[1], CartesianIndices((OX, OY)))
Bz11 = map(j -> (i = Tuple(j); @variables Bz11($i))[1], CartesianIndices((OX, OY)))
vec11 = vcat(Ex11[:], Ey11[:], Ez11[:], Bx11[:], By11[:], Bz11[:])
Ex21 = map(j -> (i = Tuple(j); @variables Ex21($i))[1], CartesianIndices((OX, OY)))
Ey21 = map(j -> (i = Tuple(j); @variables Ey21($i))[1], CartesianIndices((OX, OY)))
Ez21 = map(j -> (i = Tuple(j); @variables Ez21($i))[1], CartesianIndices((OX, OY)))
Bx21 = map(j -> (i = Tuple(j); @variables Bx21($i))[1], CartesianIndices((OX, OY)))
By21 = map(j -> (i = Tuple(j); @variables By21($i))[1], CartesianIndices((OX, OY)))
Bz21 = map(j -> (i = Tuple(j); @variables Bz21($i))[1], CartesianIndices((OX, OY)))
vec21 = vcat(Ex21[:], Ey21[:], Ez21[:], Bx21[:], By21[:], Bz21[:])
Ex31 = map(j -> (i = Tuple(j); @variables Ex31($i))[1], CartesianIndices((OX, OY)))
Ey31 = map(j -> (i = Tuple(j); @variables Ey31($i))[1], CartesianIndices((OX, OY)))
Ez31 = map(j -> (i = Tuple(j); @variables Ez31($i))[1], CartesianIndices((OX, OY)))
Bx31 = map(j -> (i = Tuple(j); @variables Bx31($i))[1], CartesianIndices((OX, OY)))
By31 = map(j -> (i = Tuple(j); @variables By31($i))[1], CartesianIndices((OX, OY)))
Bz31 = map(j -> (i = Tuple(j); @variables Bz31($i))[1], CartesianIndices((OX, OY)))
vec31 = vcat(Ex31[:], Ey31[:], Ez31[:], Bx31[:], By31[:], Bz31[:])

Ex12 = map(j -> (i = Tuple(j); @variables Ex12($i))[1], CartesianIndices((OX, OY)))
Ey12 = map(j -> (i = Tuple(j); @variables Ey12($i))[1], CartesianIndices((OX, OY)))
Ez12 = map(j -> (i = Tuple(j); @variables Ez12($i))[1], CartesianIndices((OX, OY)))
Bx12 = map(j -> (i = Tuple(j); @variables Bx12($i))[1], CartesianIndices((OX, OY)))
By12 = map(j -> (i = Tuple(j); @variables By12($i))[1], CartesianIndices((OX, OY)))
Bz12 = map(j -> (i = Tuple(j); @variables Bz12($i))[1], CartesianIndices((OX, OY)))
vec12 = vcat(Ex12[:], Ey12[:], Ez12[:], Bx12[:], By12[:], Bz12[:])
Ex22 = map(j -> (i = Tuple(j); @variables Ex22($i))[1], CartesianIndices((OX, OY)))
Ey22 = map(j -> (i = Tuple(j); @variables Ey22($i))[1], CartesianIndices((OX, OY)))
Ez22 = map(j -> (i = Tuple(j); @variables Ez22($i))[1], CartesianIndices((OX, OY)))
Bx22 = map(j -> (i = Tuple(j); @variables Bx22($i))[1], CartesianIndices((OX, OY)))
By22 = map(j -> (i = Tuple(j); @variables By22($i))[1], CartesianIndices((OX, OY)))
Bz22 = map(j -> (i = Tuple(j); @variables Bz22($i))[1], CartesianIndices((OX, OY)))
vec22 = vcat(Ex22[:], Ey22[:], Ez22[:], Bx22[:], By22[:], Bz22[:])
Ex32 = map(j -> (i = Tuple(j); @variables Ex32($i))[1], CartesianIndices((OX, OY)))
Ey32 = map(j -> (i = Tuple(j); @variables Ey32($i))[1], CartesianIndices((OX, OY)))
Ez32 = map(j -> (i = Tuple(j); @variables Ez32($i))[1], CartesianIndices((OX, OY)))
Bx32 = map(j -> (i = Tuple(j); @variables Bx32($i))[1], CartesianIndices((OX, OY)))
By32 = map(j -> (i = Tuple(j); @variables By32($i))[1], CartesianIndices((OX, OY)))
Bz32 = map(j -> (i = Tuple(j); @variables Bz32($i))[1], CartesianIndices((OX, OY)))
vec32 = vcat(Ex32[:], Ey32[:], Ez32[:], Bx32[:], By32[:], Bz32[:])

Ex13 = map(j -> (i = Tuple(j); @variables Ex13($i))[1], CartesianIndices((OX, OY)))
Ey13 = map(j -> (i = Tuple(j); @variables Ey13($i))[1], CartesianIndices((OX, OY)))
Ez13 = map(j -> (i = Tuple(j); @variables Ez13($i))[1], CartesianIndices((OX, OY)))
Bx13 = map(j -> (i = Tuple(j); @variables Bx13($i))[1], CartesianIndices((OX, OY)))
By13 = map(j -> (i = Tuple(j); @variables By13($i))[1], CartesianIndices((OX, OY)))
Bz13 = map(j -> (i = Tuple(j); @variables Bz13($i))[1], CartesianIndices((OX, OY)))
vec13 = vcat(Ex13[:], Ey13[:], Ez13[:], Bx13[:], By13[:], Bz13[:])
Ex23 = map(j -> (i = Tuple(j); @variables Ex23($i))[1], CartesianIndices((OX, OY)))
Ey23 = map(j -> (i = Tuple(j); @variables Ey23($i))[1], CartesianIndices((OX, OY)))
Ez23 = map(j -> (i = Tuple(j); @variables Ez23($i))[1], CartesianIndices((OX, OY)))
Bx23 = map(j -> (i = Tuple(j); @variables Bx23($i))[1], CartesianIndices((OX, OY)))
By23 = map(j -> (i = Tuple(j); @variables By23($i))[1], CartesianIndices((OX, OY)))
Bz23 = map(j -> (i = Tuple(j); @variables Bz23($i))[1], CartesianIndices((OX, OY)))
vec23 = vcat(Ex23[:], Ey23[:], Ez23[:], Bx23[:], By23[:], Bz23[:])
Ex33 = map(j -> (i = Tuple(j); @variables Ex33($i))[1], CartesianIndices((OX, OY)))
Ey33 = map(j -> (i = Tuple(j); @variables Ey33($i))[1], CartesianIndices((OX, OY)))
Ez33 = map(j -> (i = Tuple(j); @variables Ez33($i))[1], CartesianIndices((OX, OY)))
Bx33 = map(j -> (i = Tuple(j); @variables Bx33($i))[1], CartesianIndices((OX, OY)))
By33 = map(j -> (i = Tuple(j); @variables By33($i))[1], CartesianIndices((OX, OY)))
Bz33 = map(j -> (i = Tuple(j); @variables Bz33($i))[1], CartesianIndices((OX, OY)))
vec33 = vcat(Ex33[:], Ey33[:], Ez33[:], Bx33[:], By33[:], Bz33[:])

vec = [vec11 vec12 vec13; vec21 vec22 vec23; vec31 vec32 vec33][:]

foo(x) = x
function foo(x::Rational)
  if x.den == 1
    return x.num
  else
    return x
  end
end

rationalround(x, digits=10) = Rational(round(x, digits=digits))

outputV = rationalround.(vfsm) * vec
outputS = rationalround.(sfsm) * vec

output = rationalround.(M) * vec

for i in eachindex(vec)
  for j in Ex22
    if any(vec[i] === j)
      @show vec[i], output[i]
    end
  end
end
for i in eachindex(vec)
  for j in Ey22
    if any(vec[i] === j)
      @show vec[i], output[i]
    end
  end
end
for i in eachindex(vec)
  for j in Ez22
    if any(vec[i] === j)
      @show vec[i], output[i]
    end
  end
end
for i in eachindex(vec)
  for j in Bx22
    if any(vec[i] === j)
      @show vec[i], output[i]
    end
  end
end
for i in eachindex(vec)
  for j in By22
    if any(vec[i] === j)
      @show vec[i], output[i]
    end
  end
end
for i in eachindex(vec)
  for j in Bz22
    if any(vec[i] === j)
      @show vec[i], output[i]
    end
  end
end

#@show "Full matrix operator"
#output = rationalround.(M) * vec
#for i in eachindex(output)
#  @show vec[i], output[i]
#end


