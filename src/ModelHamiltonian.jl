module ModelHamiltonian

#using ..ModelParameters
#using ..LatticeGeometry
using AltermagneticHubbardModel
using LinearAlgebra

export build_hamiltonian

"""
    build_hamiltonian(k::Tuple{Float64,Float64}, params::ModelParams, δm::Float64) -> Matrix{Float64}

Constructs the mean-field Hamiltonian matrix for given momentum k, parameters, and order parameter δm.
"""

function build_hamiltonian(k::Tuple{Float64,Float64}, params::ModelParams, δm::Float64)
    kx, ky = k
    t  = params.t
    t′ = params.t_prime
    U = params.U
    δ = params.δ

    
    if params.lattice == SQUARE
        
        # Square lattice next-nearest neighbor vectors
        a1 = [1.0, 1.0]
        a2 = [1.0, -1.0]

        # Diagonal hoppings (altermagnetic)
        t_plus  = t′ * (1 + δ)
        t_minus = t′ * (1 - δ)
        
        # Nearest-neighbor hopping
        nn_hop = -2t * (cos(kx) + cos(ky))
        
        # Sublattice diagonal terms
        hAA_up = -1.0 * ( 2t_minus * cos(dot(k,a1)) + 2t_plus  * cos(dot(k,a2))) - U*δm
        hBB_up = -1.0 * ( 2t_plus  * cos(dot(k,a1)) + 2t_minus * cos(dot(k,a2))) + U*δm

        hAA_dn = -1.0 * ( 2t_minus * cos(dot(k,a1)) + 2t_plus  * cos(dot(k,a2))) + U*δm
        hBB_dn = -1.0 * ( 2t_plus  * cos(dot(k,a1)) + 2t_minus * cos(dot(k,a2))) - U*δm
        
        H = [
            hAA_up   nn_hop    0.0      0.0;
            nn_hop  hBB_up     0.0      0.0;
            0.0      0.0      hAA_dn   nn_hop;
            0.0      0.0      nn_hop  hBB_dn
        ]
        
    elseif params.lattice == HONEYCOMB

       # Honeycomb lattice geometry (a=1 units)
        # Nearest-neighbor vectors (A -> B)
        δ₁ = [ 0.0,  1/√3 ]
        δ₂ = [-1/2, -1/(2√3)]
        δ₃ = [ 1/2, -1/(2√3)]
        NN_vectors = [δ₁, δ₂, δ₃]
        
        # Second-neighbor vectors (A->A or B->B)
        # These are the three possible directions for same-sublattice hopping
        d1 = δ₁ - δ₂  # = [1/2, √3/2] (equivalent to a1)
        d2 = δ₂ - δ₃  # = [-1, 0]
        d3 = δ₃ - δ₁  # = [1/2, -√3/2] (equivalent to a2)
        SNN_vectors = [d1, d2, d3, -d1, -d2, -d3]  # Include opposite directions
        
        # Angles (in radians) for each SNN vector relative to x-axis
        SNN_angles = [atan(d[2], d[1]) for d in SNN_vectors]  # [π/3, π, -π/3, -2π/3, 0, 2π/3]
        
        # Nearest-neighbor hopping (spin-independent)
        γk = -t * sum(exp(-1im * dot(k,δ)) for δ in NN_vectors)

        # Spin-dependent second-neighbor hopping
        function λk(σ::Int)  # σ = +1 (↑) or -1 (↓)
            return sum(
                t′ * (1 + σ*δ*cos(3θ)) * exp(-1im * dot(k, d))
                for (d,θ) in zip(SNN_vectors, SNN_angles) )
            
        end
        
        # Sublattice potential terms (altermagnetic order)
        hAA_up = -U*δm + λk(+1)
        hBB_up = +U*δm + λk(+1)
        hAA_dn = +U*δm + λk(-1)
        hBB_dn = -U*δm + λk(-1)
        
        # Build 4×4 Hamiltonian in basis (A↑, B↑, A↓, B↓)
        H = zeros(ComplexF64, 4, 4)
        H[1,1] = hAA_up
        H[2,2] = hBB_up
        H[3,3] = hAA_dn
        H[4,4] = hBB_dn
        H[1,2] = H[3,4] = γk
        H[2,1] = H[4,3] = conj(γk)
            
    elseif params.lattice == TRIANGULAR
        # Triangular lattice Hamiltonian (to be implemented)
        error("Triangular lattice not yet implemented")
    else
        error("Unknown lattice type")
    end
    
    return H
end

end # module