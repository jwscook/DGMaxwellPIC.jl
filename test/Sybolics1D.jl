using DGMaxwellPIC, Plots, TimerOutputs, StaticArrays, LinearAlgebra

const NX = 3;

const OX = 2;

const state2D = State([OX], LobattoNodes);

const DIMS = 1
const L = NX * 2;

const a = zeros(DIMS);
const b = ones(DIMS) .* L;

gridposition(x) = SVector{DIMS, Float64}((x .* (b .- a) .+ a))

const g = Grid([Cell(deepcopy(state2D), gridposition((i-1)/NX), gridposition(i/NX)) for i in 1:NX]);

const s0 = DGMaxwellPIC.speedoflight
const k = 4 * pi / L
const ω = s0 * k

fBz(x, t=0) = sin(x[1] * k - ω  * t)
fEy(x, t=0) = s0 * fBz(x, t)

electricfield!(g, fEy, 2);
magneticfield!(g, fBz, 3);
#DGMaxwellPIC.electricfielddofs!(g, s0, 2);
#DGMaxwellPIC.magneticfielddofs!(g, 1.0, 3);

const dtc = minimum((b .- a)./NX./OX) / s0
const dt = dtc * 0.2
const upwind = 1

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

vfsm = DGMaxwellPIC.volumefluxstiffnessmatrix(g)
sfsm = DGMaxwellPIC.surfacefluxstiffnessmatrix(g, upwind)

using Plots
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
Ex1 = map(i -> (@variables Ex1($i))[1], 0:OX-1)
Ey1 = map(i -> (@variables Ey1($i))[1], 0:OX-1)
Ez1 = map(i -> (@variables Ez1($i))[1], 0:OX-1)
Bx1 = map(i -> (@variables Bx1($i))[1], 0:OX-1)
By1 = map(i -> (@variables By1($i))[1], 0:OX-1)
Bz1 = map(i -> (@variables Bz1($i))[1], 0:OX-1)
vec1 = vcat(Ex1, Ey1, Ez1, Bx1, By1, Bz1)
Ex2 = map(i -> (@variables Ex2($i))[1], 0:OX-1)
Ey2 = map(i -> (@variables Ey2($i))[1], 0:OX-1)
Ez2 = map(i -> (@variables Ez2($i))[1], 0:OX-1)
Bx2 = map(i -> (@variables Bx2($i))[1], 0:OX-1)
By2 = map(i -> (@variables By2($i))[1], 0:OX-1)
Bz2 = map(i -> (@variables Bz2($i))[1], 0:OX-1)
vec2 = vcat(Ex2, Ey2, Ez2, Bx2, By2, Bz2)
Ex3 = map(i -> (@variables Ex3($i))[1], 0:OX-1)
Ey3 = map(i -> (@variables Ey3($i))[1], 0:OX-1)
Ez3 = map(i -> (@variables Ez3($i))[1], 0:OX-1)
Bx3 = map(i -> (@variables Bx3($i))[1], 0:OX-1)
By3 = map(i -> (@variables By3($i))[1], 0:OX-1)
Bz3 = map(i -> (@variables Bz3($i))[1], 0:OX-1)
vec3 = vcat(Ex3, Ey3, Ez3, Bx3, By3, Bz3)
vec = vcat(vec1, vec2, vec3)


@show "Full matrix operator"
output = M * vec
for i in eachindex(output)
  @show vec[i], output[i]
end


