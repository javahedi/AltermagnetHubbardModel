using AltermagneticHubbardModel

using Test

@testset "AltermagneticHubbardModel Tests" begin
    # Test reciprocal vectors for square lattice
    b1, b2 = get_reciprocal_vectors(SQUARE)
    @test b1 ≈ [2π, 0.0]
    @test b2 ≈ [0.0, 2π]
    
    # Test Hamiltonian construction
    params = ModelParams(SQUARE, 1.0, 0.1, 0.5, 2.0, 1.0, 10.0, 10, 0.4, 1e-6)
    H = build_hamiltonian((0.0, 0.0), params, 0.1)
    @test size(H) == (4,4)
    @test H[1,1] ≈ -2*0.1*(1-0.5)*(2) - 2.0*0.1
end