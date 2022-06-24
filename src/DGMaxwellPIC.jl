module DGMaxwellPIC

using Base.Threads
using ConcreteStructs
using FastGaussQuadrature
using FLoops
using ForwardDiff
using HCubature
using LazyArrays
using LinearAlgebra
using LoopVectorization
#using Memoize
using Memoization
using OffsetArrays
using Primes
using QuadGK
using SparseArrays
#using SparseMatricesCSR
using SpecialPolynomials
using StaticArrays
#using ThreadedSparseCSR
using ThreadsX

const speedoflight = 1.0
const epsilon0 = 1.0
const mu0 = 1.0

const ATOL=2eps()
const RTOL=2eps()

@enum FaceDirection High Low
opposite(side::FaceDirection) = side == High ? Low : High

include("Lagrange.jl")
include("States.jl")
include("Cells.jl")
include("Grids.jl")
include("DofUtilities.jl")
include("Assembly.jl")
include("ParticleData.jl")
include("Species.jl")
include("Plasma.jl")

export lagrange, LegendreNodes, LobattoNodes
export State, Cell, Grid
export electricfield, magneticfield, currentfield, chargefield
export electricfield!, magneticfield!, currentfield!, chargefield!
export facedofindices, dofs, dofs!, currentdofs
export assemble
export depositcurrent!, zerocurrent!
export sources, sources!
export Species, weight!
export Plasma, advance!

end # module

