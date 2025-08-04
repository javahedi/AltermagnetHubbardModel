module SelfConsistentLoop

using AltermagneticHubbardModel
using LinearAlgebra

export run_scf, find_chemical_potential




"""
    calculate_observables(H, μ, β, lattice)

Calculates both staggered (δm) and net (m_plus) magnetization from the Hamiltonian.
"""
function calculate_observables(H::AbstractMatrix, μ::Float64, β::Float64, lattice::Symbol)

    vals, vecs = eigen(Hermitian(H))
    n = zeros(ComplexF64, size(H))  # Ensure complex for SOC

    total_density = 0.0
    for (i,ϵ) in enumerate(vals)
        f = fermi(ϵ, μ, β)
        n += f * vecs[:,i] * vecs[:,i]'  # Outer product
        total_density += f
    end

    n_real = real(n)  # Physical observables are real

    if lattice == HEXATRIANGULAR
        # 3-sublattice altermagnetic order parameter
        n_real = real(n)

        
        
    elseif lattice == SQUARE 
        # Original 2-sublattice formula
        # (A↑, B↑, A↓, B↓)
        mA = n_real[1,1] - n_real[3,3]  # A↑ - A↓
        mB = n_real[2,2] - n_real[4,4]  # B↑ - B↓
        δm = (mA - mB) / 4             # Staggered magnetization (A-B)
        m_plus = (mA + mB) / 2         # Net magnetization
        
    elseif lattice == ALPHA_T3
         # For α-T3 lattice (A↑, B↑, C↑, A↓, B↓, C↓)
        mA = n_real[1,1] - n_real[4,4]  # A↑ - A↓
        mB = n_real[2,2] - n_real[5,5]  # B↑ - B↓
        mC = n_real[3,3] - n_real[6,6]  # C↑ - C↓
        
        δm = (mA - mB) / 4             # Staggered magnetization (A-B)
        m_plus = (mA + mB + mC) / 6    # Net magnetization (all sites)

    elseif lattice == KMmodel
        # For Kane-Mele model (A↑, B↑, A↓, B↓)
        mA = n_real[1,1] - n_real[3,3]  # A↑ - A↓
        mB = n_real[2,2] - n_real[4,4]  # B↑ - B↓

        δm = (mA - mB) / 4             # Staggered magnetization (A-B)
        m_plus = (mA + mB) / 2         # Net magnetization
    end
    return δm, m_plus, total_density/2 
end


"""
    find_chemical_potential(params, δm; μ_min, μ_max)

Find chemical potential μ that gives target density n using bisection method.
"""
function find_chemical_potential(params::ModelParams, δm::Float64; μ_min=-10.0, μ_max=10.0)
    kpoints  = generate_kpoints(params.lattice, params.kpoints)
    target_n = params.n
    
    function compute_n(μ)
        n_total = 0.0
        for k in kpoints
            H        = build_hamiltonian(k, params, δm)
            vals, _  = eigen(Hermitian(H))
            n_total += sum(fermi.(vals, μ, params.β))
        end
        
        return n_total / (2 * length(kpoints))
        
    end
    
    # Bisection search  
    while (μ_max - μ_min > params.tol)
        μ = (μ_min + μ_max) / 2
        current_n = compute_n(μ)
        if current_n < target_n
            μ_min = μ
        else
            μ_max = μ
        end
       
    end
    
    return (μ_min + μ_max) / 2
    
    
end

"""
    run_scf(params::ModelParams; verbose::Bool=true) -> Tuple{Float64, Float64}

Performs SCF calculation, returns both δm and m_plus.
"""
function run_scf(params::ModelParams; verbose::Bool=true)
    kpoints = generate_kpoints(params.lattice, params.kpoints)
    δm = 0.1  # Initial guess for staggered magnetization
    m_plus = 0.0
    iter = 0
    max_iter = 100
    μ = 0.0

    while iter < max_iter
        iter += 1
        δm_prev = δm
        
        # Find chemical potential
        μ = find_chemical_potential(params, δm)
        
        # Compute observables
        δm_new = 0.0
        m_plus_new = 0.0
        n_total = 0.0
        
        for k in kpoints
            H = build_hamiltonian(k, params, δm)
            δm_k, m_plus_k, n_k = calculate_observables(H, μ, params.β, params.lattice)
            δm_new += δm_k
            m_plus_new += m_plus_k
            n_total += n_k
        end
        
        # Average over k-points
        δm_new /= length(kpoints)
        m_plus_new /= length(kpoints)
        n_total /= length(kpoints)
        
        # Mixing
        δm = params.mixing * δm_prev + (1 - params.mixing) * δm_new
        
        verbose && println("Iter $iter: δm = ", round(δm, digits=6),
                         " | m_+ = ", round(m_plus_new, digits=10),
                         " | μ = ", round(μ, digits=6),
                         " | n = ", round(n_total, digits=6))
        
        # Check convergence
        abs(δm - δm_prev) < params.tol && break
    end
    
    if iter == max_iter
        @warn "SCF did not converge after $max_iter iterations"
    elseif verbose
        println("SCF converged in $iter iterations")
    end
    
    return δm, m_plus, μ
end

end # module