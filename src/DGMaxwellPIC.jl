module DGMaxwellPIC

using SpecialPolynomials, QuadGK, OffsetArrays, SparseArrays, Memoization, LinearAlgebra
using StaticArrays, ConcreteStructs, FastGaussQuadrature
const speedoflight = 1.0
const epsilon0 = 1.0

abstract type BasisFunctionType end
abstract type Lagrange <: BasisFunctionType end
struct LagrangeOrthogonal <: Lagrange end
#struct Chebyshev <: BasisFunctionType end # implement later if necessary
#struct Legendre <: BasisFunctionType end # implement later if necessary

@enum FaceDirection High Low

include("States.jl")
include("Cells.jl")
include("Grids.jl")
include("DofUtilities.jl")
include("Assembly.jl")
include("Lagrange.jl")


function sources(g::Grid)
  A = volumemassmatrix(g)
  x = sparsevec(1:currentndofs(g), currentdofs(g), ndofs(g))
  return A * x
end

function depositcurrent(g::Grid, x, current)

end

export LagrangeOrthogonal
export lagrange
export State, Cell, Grid
export electricfield, magneticfield, currentfield, chargefield
export electricfield!, magneticfield!, currentfield!, chargefield!
export facedofindices, dofs
export assemble
export depositcurrent
export sources

end # module

#=
const NX = 4
const NY = 3
const OX = 3
const OY = 2

state2D = State([OX, OY], LagrangeOrthogonal)
magneticfielddofs!(state2D, 3, 1.0)

grid2D = Grid([Cell(deepcopy(state2D), ((i-1)/NX, (j-1)/NY), (i/NX, j/NY)) for i in 1:NX, j in 1:NY])

=#
