module DGMaxwellPIC

using ConcreteStructs
using FastGaussQuadrature
using ForwardDiff
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
include("ParticleData.jl")
include("ParticlePushers.jl")
include("Species.jl")
include("Plasma.jl")

export LagrangeOrthogonal
export lagrange
export State, Cell, Grid
export electricfield, magneticfield, currentfield, chargefield
export electricfield!, magneticfield!, currentfield!, chargefield!
export facedofindices, dofs, dofs!, currentdofs
export assemble
export depositcurrent!
export sources
export Species, weight!
export Plasma

end # module

