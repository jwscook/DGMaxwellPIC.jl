using DGMaxwellPIC, ForwardDiff, QuadGK, Suppressor, Test

import DGMaxwellPIC: lagrange, volumemassmatrix, massmatrix, NDimNodes, volumefluxstiffnessmatrix

@testset "Massmatrices" begin

  minorder = 2
  maxorder = 6
  
  legendredict = Dict()
  lobattodict = Dict()
  for n in minorder:maxorder
    legendrenodes = NDimNodes((n,), LegendreNodes)
    lobattonodes = NDimNodes((n,), LobattoNodes)
    for i in 1:n, j in 1:n
      legendredict[(n, i, j, 0)] = quadgk(x->lagrange(x, legendrenodes[1], i) *
        lagrange(x, legendrenodes[1], j), -1, 1, atol=100eps())[1]
      legendredict[(n, i, j, 1)]  = quadgk(x->lagrange(x, legendrenodes[1], i) *
        ForwardDiff.derivative(y->lagrange(y, legendrenodes[1], j), x),
         -1, 1, atol=100eps())[1]
      if i != j
        @test legendredict[(n, i, j, 0)] ≈ 0.0 atol=1000eps()
        legendredict[(n, i, j, 0)] = 0.0
      end
      lobattodict[(n, i, j, 0)] = quadgk(x->lagrange(x, lobattonodes[1], i) *
        lagrange(x, lobattonodes[1], j), -1, 1, atol=100eps())[1]
      lobattodict[(n, i, j, 1)]  = quadgk(x->lagrange(x, lobattonodes[1], i) *
        ForwardDiff.derivative(y->lagrange(y, lobattonodes[1], j), x), -1, 1, atol=100eps())[1]
    end
  end

  function recalcmassmatrix(dict::Dict, orders::Tuple, i::Integer, j::Integer, derivs=zeros(length(orders)))
    dofsize = prod(orders)
    cart = CartesianIndices(orders)
    ii, jj = Tuple(cart[i]), Tuple(cart[j])
    output = 1.0
    for (c, n) in enumerate(orders)
      output *= dict[(n, ii[c], jj[c], derivs[c])]
    end
    return output
  end

  for DIMS in (1, 2, 3)
    @testset "($DIMS)D" begin
      for (NodeType, dict) in ((LegendreNodes, legendredict), (LobattoNodes, lobattodict)), counter in 1:3
        orders = Tuple(rand(minorder:maxorder, DIMS))
        nodes = NDimNodes(orders, NodeType)
        mmreference = @suppress massmatrix(nodes, nodes)
        for i in 1:size(mmreference, 1), j in 1:size(mmreference, 2)
          @test mmreference[i, j] ≈ recalcmassmatrix(dict, orders, i, j) atol=1000eps()
        end
        derivdim = rand(1:DIMS)
        vfmmreference = volumefluxstiffnessmatrix(nodes, nodes, derivdim)
        for i in 1:size(vfmmreference, 1), j in 1:size(vfmmreference, 2)
          @test vfmmreference[i, j] ≈ recalcmassmatrix(dict, orders, i, j, [i == derivdim for i in 1:DIMS]) atol=1000eps()
        end
      end
    end
  end

end
