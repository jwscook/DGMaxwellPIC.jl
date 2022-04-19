using DGMaxwellPIC, FastGaussQuadrature, ForwardDiff, Polynomials, SpecialPolynomials, Test

import DGMaxwellPIC: lagrange, lagrangederiv

@testset "Lagrange" begin
  for (NodeType, pointsweightsfun) in ((LegendreNodes, gausslegendre), (LobattoNodes, gausslobatto))
    @testset "$NodeType" begin
      for n in 2:9
        nodes = NodeType(n)
        points, weights = pointsweightsfun(n)
        for i in 1:n
          coeffs = [k == i for k in 1:n]
          poly = SpecialPolynomials.Lagrange(points, coeffs)
          for j in 1:10
            x = rand() * 2 - 1
            @test nodes(x, i) ≈ poly(x)
            @test lagrange(x, nodes, i) ≈ poly(x)
            @test lagrangederiv(x, nodes, i) ≈ ForwardDiff.derivative(poly, x)
          end
        end
      end
    end 
  end
end
