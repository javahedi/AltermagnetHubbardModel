module SelfConsistentLoop

using AltermagneticHubbardModel
using LinearAlgebra

export run_scf, find_chemical_potential, fermi


"""
    fermi(ϵ, μ, β)

Fermi-Dirac distribution function.
"""
function fermi(ϵ::Float64, μ::Float64, β::Float64)
    1.0 / (exp(β * (ϵ - μ)) + 1.0)
end



function calculate_observables(H::AbstractMatrix, μ::Float64, β::Float64, lattice::Symbol)
    vals, vecs = eigen(Hermitian(H))
    n = zeros(ComplexF64, size(H))  # Ensure complex for SOC

    total_density = 0.0
    for (i,ϵ) in enumerate(vals)
        f = fermi(ϵ, μ, β)
        n += f * vecs[:,i] * vecs[:,i]'  # Outer product
        total_density += f
    end
    
    if lattice == HEXATRIANGULAR
        # 3-sublattice altermagnetic order parameter
        n_real = real(n)
        # Proper 3-sublattice altermagnetic order
        δm = (n_real[1,1] - n_real[2,2] + n_real[3,3] - n_real[4,4] + n_real[5,5] - n_real[6,6]) / 6
    elseif lattice == SQUARE 
        # Original 2-sublattice formula
        # (A↑, B↑, A↓, B↓)
        n_real = real(n)
        δm = (n_real[1,1] - n_real[2,2] - n_real[3,3] + n_real[4,4]) / 4
    elseif lattice == ALPHA_T3
        # (A↑, B↑, C↑, A↓, B↓, C↓)
        n_real = real(n)
        # δm = (m_A - m_B) / 2
        mA = n_real[1,1] - n_real[4,4]  # spin↑ - spin↓ on A
        mB = n_real[2,2] - n_real[5,5]  # spin↑ - spin↓ on B
        δm = (mA - mB) / 4

        #m_total = sum(real(n[i,i]) - real(n[i+3,i+3]) for i in 1:3)  # A, B, C

    end
    return δm, total_density/2 #, m_total
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
    run_scf(params::ModelParams; verbose::Bool=true) -> Float64

Performs self-consistent field calculation to determine the altermagnetic order parameter δm.
"""
function run_scf(params::ModelParams; verbose::Bool=true)
    kpoints = generate_kpoints(params.lattice, params.kpoints)
    δm = 0.1  # Initial guess
    δm_prev = 0.0
    iter = 0
    max_iter = 100
    
    while abs(δm - δm_prev) > params.tol && iter < max_iter
        iter += 1
        δm_prev = δm
        
        # Find chemical potential
        μ = find_chemical_potential(params, δm)
        
        # Compute new δm
        δm_new = 0.0
        n_total = 0.0
        for k in kpoints
            H         = build_hamiltonian(k, params, δm)
            δm_k, n_k  = calculate_observables(H, μ, params.β, params.lattice)
            δm_new += δm_k
            n_total += n_k
        end
        δm_new  /= length(kpoints)
        n_total /= length(kpoints)
        
        # Mixing
        δm = real(params.mixing * δm_prev + (1 - params.mixing) * δm_new)

        
        verbose && println("Iter $iter: δm = ", round(δm, digits=6), 
                         ", μ = ", round(μ, digits=6),
                         ", n = ", round(n_total, digits=6))
    end
    
    if iter == max_iter
        @warn "SCF did not converge after $max_iter iterations"
    elseif verbose
        println("SCF converged in $iter iterations")
    end
    
    return δm
end

end # module