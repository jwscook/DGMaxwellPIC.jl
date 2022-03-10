module DGMaxwellPIC

using ConcreteStructs
using FastGaussQuadrature
using HaltonSequences
using HCubature
using LinearAlgebra
using Memoization
using OffsetArrays
using Primes
using QuadGK
using SparseArrays
using SpecialPolynomials
using StaticArrays

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
include("Particles.jl")

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

