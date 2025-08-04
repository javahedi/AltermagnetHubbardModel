
function fermi(ϵ::Float64, μ::Float64, β::Float64)
    1.0 / (exp(β * (ϵ - μ)) + 1.0)
end



function calculate_net_magnetization(H_up::AbstractMatrix, H_dn::AbstractMatrix, μ::Float64, β::Float64, N::Int)
    """
    Computes the net magnetization m_+ = (1/2N) ∑ᵢ [⟨n_{iA↑}⟩ + ⟨n_{iB↑}⟩ - ⟨n_{iA↓}⟩ - ⟨n_{iB↓}⟩].
    
    Args:
        H_up, H_dn : Mean-field Hamiltonians for spin-up and spin-down sectors.
        μ          : Chemical potential (for Fermi-Dirac distribution).
        β          : Inverse temperature (β = 1/(k_B T)).
        N          : Number of unit cells.
    
    Returns:
        m_plus     : Net magnetization per site.
    """
    # Diagonalize Hamiltonians
    vals_up, vecs_up = eigen(Hermitian(H_up))
    vals_dn, vecs_dn = eigen(Hermitian(H_dn))
    
    # Initialize densities
    n_A_up = 0.0
    n_B_up = 0.0
    n_A_dn = 0.0
    n_B_dn = 0.0
    
    # Compute occupations for spin-up
    for (i, ϵ) in enumerate(vals_up)
        f = fermi(ϵ, μ, β)  # Fermi-Dirac distribution
        n_A_up += f * abs2(vecs_up[1, i])  # A-sublattice (1st component in spinor)
        n_B_up += f * abs2(vecs_up[2, i])  # B-sublattice (2nd component)
    end
    
    # Compute occupations for spin-down
    for (i, ϵ) in enumerate(vals_dn)
        f = fermi(ϵ, μ, β)
        n_A_dn += f * abs2(vecs_dn[1, i])
        n_B_dn += f * abs2(vecs_dn[2, i])
    end
    
    # Net magnetization per site
    m_plus = (n_A_up + n_B_up - n_A_dn - n_B_dn) / (2 * N)
    return m_plus
end