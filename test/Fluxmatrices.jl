using DGMaxwellPIC, ForwardDiff, QuadGK, Suppressor, Symbolics, Test

import DGMaxwellPIC: lagrange, volumemassmatrix, massmatrix, NDimNodes, surfacefluxstiffnessmatrix, Low, High

@testset "Fluxmatrices" begin

  @testset "1D" begin
    DIMS = 1
    for OX in (2, 3)

      Ex = map(i -> (@variables Ex($i))[1], 1:OX)
      Ey = map(i -> (@variables Ey($i))[1], 1:OX)
      Ez = map(i -> (@variables Ez($i))[1], 1:OX)
      Bx = map(i -> (@variables Bx($i))[1], 1:OX)
      By = map(i -> (@variables By($i))[1], 1:OX)
      Bz = map(i -> (@variables Bz($i))[1], 1:OX)
      vec = vcat(Ex, Ey, Ez, Bx, By, Bz)


      for NodeType in (LegendreNodes, LobattoNodes)
        nodes = NDimNodes(Tuple(OX for _ in 1:DIMS), NodeType)
        state1D = State(OX .* ones(Int64, DIMS), NodeType)
        cell = Cell(state1D, -ones(DIMS), ones(DIMS)) # in reference cell
        sfmm = surfacefluxstiffnessmatrix(cell, nodes, 1, Low)
        @test length(vec) == size(sfmm, 2)
        @show DIMS, OX, NodeType, Low
        #display("text/plain", Matrix(sfmm))
        o = trunc.(sfmm, digits=5) * vec
        for i in eachindex(o)
          @show i, vec[i], o[i]
        end
        sfmm = surfacefluxstiffnessmatrix(cell, nodes, 1, High)
        @test length(vec) == size(sfmm, 2)
        @show DIMS, OX, NodeType, High
        #display("text/plain", Matrix(sfmm))
        o = trunc.(sfmm, digits=5) * vec
        for i in eachindex(o)
          @show i, vec[i], o[i]
        end
      end
    end

  end

  @testset "2D" begin
    DIMS = 2
    for OX in (2, 3), OY in (2:3)

      Ex = map(j -> (i = Tuple(j); @variables Ex($i))[1], CartesianIndices((OX, OY)))
      Ey = map(j -> (i = Tuple(j); @variables Ey($i))[1], CartesianIndices((OX, OY)))
      Ez = map(j -> (i = Tuple(j); @variables Ez($i))[1], CartesianIndices((OX, OY)))
      Bx = map(j -> (i = Tuple(j); @variables Bx($i))[1], CartesianIndices((OX, OY)))
      By = map(j -> (i = Tuple(j); @variables By($i))[1], CartesianIndices((OX, OY)))
      Bz = map(j -> (i = Tuple(j); @variables Bz($i))[1], CartesianIndices((OX, OY)))
      vec = vcat(Ex[:], Ey[:], Ez[:], Bx[:], By[:], Bz[:])

      for NodeType in (LegendreNodes, LobattoNodes)
        nodes = NDimNodes((OX, OY), NodeType)
        state1D = State([OX, OY], NodeType)
        cell = Cell(state1D, -ones(DIMS), ones(DIMS)) # in reference cell
        sfmm = surfacefluxstiffnessmatrix(cell, nodes, 1, Low)
        @test length(vec) == size(sfmm, 2)
        @show DIMS, OX, NodeType, Low
        #display("text/plain", Matrix(sfmm))
        o = trunc.(sfmm, digits=5) * vec
        for i in eachindex(o)
          @show i, vec[i], o[i]
        end
        sfmm = surfacefluxstiffnessmatrix(cell, nodes, 1, High)
        @test length(vec) == size(sfmm, 2)
        @show DIMS, OX, NodeType, High
        #display("text/plain", Matrix(sfmm))
        o = trunc.(sfmm, digits=5) * vec
        for i in eachindex(o)
          @show i, vec[i], o[i]
        end
      end
    end

  end

end
