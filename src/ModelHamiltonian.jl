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
    #μ = params.μ
    n = params.n

    
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

        # Lattice vectors
        a1 = [1.0, 0.0]
        a2 = [1/2, √3/2]
        a3 = -a1 - a2   # third NN direction in triangular lattice

        

        # Parameters
        t1 = params.t1           # NN hopping amplitude
        t2 = params.t2           # NNN hopping amplitude
        U  = params.U            # Hubbard interaction strength
        J  = params.J            # Hund/flavor exchange coupling

        

        # Flavor coherence order parameters for exchange J (off-diagonal)
        # Assume params.chi is a 3x3 Hermitian matrix of flavor coherence MF parameters
        chi = params.chi  

        # Diagonal terms: onsite + NNN hoppings
        # NNN hoppings may be direction-dependent for flavor-selective symmetry breaking
        h_diag = zeros(3)
        h_diag[1] = -2 * t2 * (cos(dot(k, a2)) + cos(dot(k, a3))) - U * δm[1]
        h_diag[2] = -2 * t2 * (cos(dot(k, a3)) + cos(dot(k, a1))) - U * δm[2]
        h_diag[3] = -2 * t2 * (cos(dot(k, a1)) + cos(dot(k, a2))) - U * δm[3]

        # Nearest neighbor hoppings (flavor off-diagonal)
        nn_hops = Dict(
            (1,2) => -t1 * (exp(im * dot(k, a1)) + exp(-im * dot(k, a1))),
            (2,3) => -t1 * (exp(im * dot(k, a2)) + exp(-im * dot(k, a2))),
            (3,1) => -t1 * (exp(im * dot(k, a3)) + exp(-im * dot(k, a3)))
        )

        # Initialize Hamiltonian
        H = zeros(ComplexF64, 3, 3)

        # Fill diagonal
        for i in 1:3
            H[i,i] = h_diag[i]
        end

        # Fill off-diagonal NN hopping + flavor exchange (J) mean-field terms
        for (i,j) in keys(nn_hops)
            H[i,j] = nn_hops[(i,j)] + J * chi[i,j]
            H[j,i] = conj(H[i,j])
        end

    elseif params.lattice == ALPHA_T3

        # Define the three sublattices: A, B, C
        # and NN hoppings: A–B (with cos(α) t), B–C (with sin(α) t)
        # Position vectors in momentum space (relative, assuming unit cell with A at origin)
        
        δAB = [[0.0, -1.0], [√3/2, 0.5], [-√3/2, 0.5]]
        δBC = [[0.0, 1.0], [-√3/2, -0.5], [√3/2, -0.5]]

       

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

    elseif params.lattice == KMmodel
        # Phys. Rev. Lett. 133, 086503 (2024)

        μ = 0.0
        # Define nearest-neighbor (NN) vectors δ_i (A→B)
        δ1 = [0.0, -1.0]
        δ2 = [√3/2, 0.5]
        δ3 = [-√3/2, 0.5]
        δs = [δ1, δ2, δ3]

        # NN structure factor f(k)
        f_k = sum(exp(im * dot(k, δ)) for δ in δs)

        # Next-nearest-neighbor (NNN) vectors a_j
        # Note: These connect sites on the *same* sublattice
        a1 = δ2 - δ3
        a2 = δ3 - δ1
        a3 = δ1 - δ2
        a_vectors = [a1, a2, a3]

        # α(k): spin-splitting term from SOC
        α_k = -2λ * sum(sin(dot(k, a)) for a in a_vectors)

        # β(k): real part of NN hopping (τ^x)
        β_k = -t * sum(cos(dot(k, δ)) for δ in δs)

        # c(k): imaginary part of NN hopping (τ^y)
        c_k = -t * sum(sin(dot(k, δ)) for δ in δs)

        # Pauli matrices
        σ₀ = Matrix{Float64}(I, 2, 2)
        σz = [1.0 0.0; 0.0 -1.0]
        
        # τ Pauli matrices (sublattice space)
        τ₀ = Matrix{Float64}(I, 2, 2)
        τx = [0.0 1.0; 1.0 0.0]
        τy = [0.0 -im; im 0.0]

        # Term 1: -μ * I₄
        Hμ = -μ * Matrix{ComplexF64}(I, 4, 4)

        # Term 2: α(k) * σz ⊗ τ₀
        Hα = kron(σz, τ₀) * α_k

        # Term 3: β(k) * σ₀ ⊗ τx
        Hβ = kron(σ₀, τx) * β_k

        # Term 4: c(k) * σ₀ ⊗ τy
        Hc = kron(σ₀, τy) * c_k

        # Total Hamiltonian
        H = Hμ + Hα + Hβ + Hc


    elseif params.lattice == TRIANGULAR
        # Triangular lattice Hamiltonian (to be implemented)
        error("Triangular lattice not yet implemented")
    else
        error("Unknown lattice type")
    end
    
    return H
end

end # module