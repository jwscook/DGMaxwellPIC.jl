using DGMaxwellPIC, FastGaussQuadrature, ForwardDiff, Polynomials, SpecialPolynomials, Test

import DGMaxwellPIC: lagrange, lagrangederiv

@testset "Grids" begin
  for (order, rtol) in ((5, 1e-1), (15, 1e-2)), NodeType in (LobattoNodes, LegendreNodes)
    _state = State([order for _ in 1:1], NodeType);
    NX = 2^rand(6:10)
    a = rand(1)
    b = a .+ abs.(randn(1)) * 10
    L = b[1] - a[1]
    gridposition(x) = x .* (b .- a) .+ a
    _grid = Grid([Cell(deepcopy(_state), gridposition((i-1)/NX), gridposition(i/NX)) for i in 1:NX]);
    k = 4 * pi / L
    ω = DGMaxwellPIC.speedoflight * k
    fBy(x, t=0) = sin(x[1] * k - ω  * t)
    magneticfield!(_grid, fBy, 2);
    x = DGMaxwellPIC.cellcentres(_grid)

    result = [magneticfield(_grid, xi, 2) for xi in x]
    expected = fBy.(x)
    @test isapprox(result, expected, rtol=rtol, atol=0.01)
  end
end
