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
    λ = params.λ
    α = params.α

    
    if params.lattice == SQUARE
        
        # Square lattice next-nearest neighbor vectors
        a1 = [1.0, 1.0]
        a2 = [1.0, -1.0]

        # Diagonal hoppings (altermagnetic)
        t_plus  = t′ * (1 + δ)
        t_minus = t′ * (1 - δ)
        
        # Nearest-neighbor hopping
        nn_hop_val = -2t * (cos(kx) + cos(ky))
        
        # Sublattice diagonal terms
        hAA_up = -1.0 * ( 2t_minus * cos(dot(k,a1)) + 2t_plus  * cos(dot(k,a2))) - U*δm
        hBB_up = -1.0 * ( 2t_plus  * cos(dot(k,a1)) + 2t_minus * cos(dot(k,a2))) + U*δm

        hAA_dn = -1.0 * ( 2t_minus * cos(dot(k,a1)) + 2t_plus  * cos(dot(k,a2))) + U*δm
        hBB_dn = -1.0 * ( 2t_plus  * cos(dot(k,a1)) + 2t_minus * cos(dot(k,a2))) - U*δm
        
        # Rashba spin-flipping terms for nearest-neighbor hopping
        # These terms connect spin-up to spin-down (and vice-versa)
        # The factor of 2 comes from the sum over +/-x and +/-y directions in Fourier transform
        rashba_coupling_term_plus  = λ * (sin(ky) + 1im * sin(kx)) # for c_up^dagger c_down
        rashba_coupling_term_minus = λ * (sin(ky) - 1im * sin(kx)) # for c_down^dagger c_up (h.c.)

        # Initialize H as a complex matrix because of Rashba SOC
        H = zeros(ComplexF64, 4, 4)

        # Populate the Hamiltonian matrix in basis (A↑, B↑, A↓, B↓)
        # 1: A↑, 2: B↑, 3: A↓, 4: B↓

        # --- Spin-up block (top-left 2x2) ---
        H[1,1] = hAA_up # A↑-A↑ NNN hopping + MF
        H[2,2] = hBB_up # B↑-B↑ NNN hopping + MF
        H[1,2] = nn_hop_val # A↑-B↑ NN hopping (spin-conserving)
        H[2,1] = nn_hop_val # B↑-A↑ NN hopping (spin-conserving)

        # --- Spin-down block (bottom-right 2x2) ---
        H[3,3] = hAA_dn # A↓-A↓ NNN hopping + MF
        H[4,4] = hBB_dn # B↓-B↓ NNN hopping + MF
        H[3,4] = nn_hop_val # A↓-B↓ NN hopping (spin-conserving)
        H[4,3] = nn_hop_val # B↓-A↓ NN hopping (spin-conserving)

        # --- Spin-mixing blocks (off-diagonal 2x2 blocks) due to Rashba SOC ---
        # These terms connect A↑ to B↓, B↑ to A↓, etc. (NN hoppings)
        
        # A↑ to B↓ (c_A_up^dagger c_B_dn)
        H[1,4] = rashba_coupling_term_plus
        # B↓ to A↑ (c_B_dn^dagger c_A_up) - Hermitian conjugate of H[1,4]
        H[4,1] = rashba_coupling_term_minus 

        # B↑ to A↓ (c_B_up^dagger c_A_dn)
        H[2,3] = rashba_coupling_term_plus
        # A↓ to B↑ (c_A_dn^dagger c_B_up) - Hermitian conjugate of H[2,3]
        H[3,2] = rashba_coupling_term_minus
        
        
    elseif params.lattice == HEXATRIANGULAR

        # HEXATRIANGULAR lattice geometry next-nearest neighbor vectors (a=1 units)
        a1 = [cos(π/3), sin(π/3)]
        a2 = [0.0, 1.0]


        # Nearest-neighbor vectors (A->B, B->C, C->A)
        δ₁ = (a1 + a2)/3
        δ₂ = (-a1 + a2)/3 
        δ₃ = (-2a1 - a2)/3
        NN_vectors = [δ₁, δ₂, δ₃]
        
        # Second-neighbor vectors (same sublattice hopping)
        d1 = a1    # A->A
        d2 = a2    # B->B
        d3 = a1 - a2  # C->C
        SNN_vectors = [d1, d2, d3, -d1, -d2, -d3]  # Include opposite directions

       

        # Build 6×6 Hamiltonian in basis (A↑, B↑, C↑, A↓, B↓, C↓)
        H = zeros(ComplexF64, 6, 6)
    
        # Diagonal terms (sublattice potential and Zeeman splitting)
        hAA_up = -2t′ * sum(cos(dot(k,d)) for d in (d1, -d1)) - U*δm
        hBB_up = -2t′ * sum(cos(dot(k,d)) for d in (d2, -d2)) - U*δm
        hCC_up = -2t′ * sum(cos(dot(k,d)) for d in (d3, -d3)) - U*δm
        
        hAA_dn = -2t′ * sum(cos(dot(k,d)) for d in (d1, -d1)) + U*δm
        hBB_dn = -2t′ * sum(cos(dot(k,d)) for d in (d2, -d2)) + U*δm
        hCC_dn = -2t′ * sum(cos(dot(k,d)) for d in (d3, -d3)) + U*δm


        # Off-diagonal terms (nearest-neighbor hopping)
        γk = -t * sum(exp(-1im * dot(k,δ)) for δ in NN_vectors)
        
        # Construct Hamiltonian blocks
        # Spin-up block (1:3, 1:3)
        H[1,1] = hAA_up
        H[2,2] = hBB_up
        H[3,3] = hCC_up
        H[1,2] = γk  # A↑-B↑ hopping
        H[2,1] = conj(γk)
        H[2,3] = γk  # B↑-C↑ hopping
        H[3,2] = conj(γk)
        H[1,3] = γk  # A↑-C↑ hopping
        H[3,1] = conj(γk)
    
        # Spin-down block (4:6, 4:6)
        H[4,4] = hAA_dn
        H[5,5] = hBB_dn
        H[6,6] = hCC_dn
        H[4,5] = γk  # A↓-B↓ hopping
        H[5,4] = conj(γk)
        H[5,6] = γk  # B↓-C↓ hopping
        H[6,5] = conj(γk)
        H[4,6] = γk  # A↓-C↓ hopping
        H[6,4] = conj(γk)

    elseif params.lattice == ALPHA_T3

        # Define the three sublattices: A, B, C
        # and NN hoppings: A–B (with cos(α) t), B–C (with sin(α) t)
        # Position vectors in momentum space (relative, assuming unit cell with A at origin)
        δAB = [[1.0, 0.0], [-0.5, √3/2], [-0.5, -√3/2]]
        δBC = [[-1.0, 0.0], [0.5, -√3/2], [0.5, √3/2]]

        # Initialize 6x6 Hamiltonian: basis = (A↑, B↑, C↑, A↓, B↓, C↓)
        H = zeros(ComplexF64, 6, 6)

        # Spin-conserving hopping: γ_AB and γ_BC
        γ_AB = -t * cos(α) * sum(exp(-1im * dot(k, d)) for d in δAB)
        γ_BC = -t * sin(α) * sum(exp(-1im * dot(k, d)) for d in δBC)

        # Mean-field onsite terms with Hubbard U on A and B
        hA_up = -U * δm
        hB_up = U * δm
        hC_up = 0.0

        hA_dn = U * δm
        hB_dn = -U * δm
        hC_dn = 0.0

        # Fill in Spin-up block
        H[1,1] = hA_up
        H[2,2] = hB_up
        H[3,3] = hC_up
        H[1,2] = γ_AB
        H[2,1] = conj(γ_AB)
        H[2,3] = γ_BC
        H[3,2] = conj(γ_BC)

        # Fill in Spin-down block
        H[4,4] = hA_dn
        H[5,5] = hB_dn
        H[6,6] = hC_dn
        H[4,5] = γ_AB
        H[5,4] = conj(γ_AB)
        H[5,6] = γ_BC
        H[6,5] = conj(γ_BC)


            
    elseif params.lattice == TRIANGULAR
        # Triangular lattice Hamiltonian (to be implemented)
        error("Triangular lattice not yet implemented")
    else
        error("Unknown lattice type")
    end
    
    return H
end

end # module