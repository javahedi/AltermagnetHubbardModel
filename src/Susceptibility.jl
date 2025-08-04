module Susceptibility

    using Distributed
    @everywhere using LinearAlgebra
    @everywhere using AltermagneticHubbardModel

    export compute_chi0_and_Uc, compute_rpa_susceptibility

    @everywhere function compute_rpa_susceptibility_at_qω(
        q::Vector{Float64},
        ωlist::Vector{Float64},
        params::ModelParams,
        μ::Float64,
        δm::Float64,
        U::Float64,
        χtype::Symbol;
        η::Float64 = 0.01,
        reduction::Function = χ -> maximum(real.(eigvals(Hermitian(χ))))
    )
        n_sublattices = (params.lattice == ALPHA_T3) ? 3 : 2
        n_spin = 2
        dim = n_sublattices * n_spin
        kpoints = generate_kpoints(params.lattice, params.kpoints)

        χω_scalar = Float64[]

        for ω in ωlist
            χ0 = zeros(ComplexF64, n_sublattices, n_sublattices)

            for k in kpoints
                H_k  = build_hamiltonian(k, params, δm)
                H_kq = build_hamiltonian(k .+ q, params, δm)
                E_k, U_k   = eigen(Hermitian(H_k))
                E_kq, U_kq = eigen(Hermitian(H_kq))

                for ν in 1:dim, ν′ in 1:dim
                    fν  = fermi(E_k[ν], μ, params.β)
                    fν′ = fermi(E_kq[ν′], μ, params.β)
                    denom = ω + E_k[ν] - E_kq[ν′] + im * η

                    spin_ν, sub_ν   = divrem(ν - 1, n_sublattices)
                    spin_ν′, sub_ν′ = divrem(ν′ - 1, n_sublattices)
                    contrib = conj(U_k[:, ν][ν]) * U_kq[:, ν′][ν′] * (fν - fν′) / denom

                    if χtype == :zz && spin_ν == spin_ν′
                        χ0[sub_ν + 1, sub_ν′ + 1] += contrib
                    elseif χtype == :pm && spin_ν == 0 && spin_ν′ == 1
                        χ0[sub_ν + 1, sub_ν′ + 1] += contrib
                    end
                end
            end

            χ0 ./= length(kpoints)
            χrpa = χ0 / (I - U * χ0)
            push!(χω_scalar, imag(reduction(χrpa)))
        end

        return ωlist, χω_scalar  # each is Vector{Float64}
    end


    
    @everywhere function compute_chi0_for_q(q, params::ModelParams, μ::Float64, δm::Float64, η=0.01)
        if params.lattice == ALPHA_T3
            n_sublattices = 3
        elseif params.lattice == SQUARE || params.lattice == HONEYCOMB || params.lattice == KMmodel
            n_sublattices = 2
        else
            error("Unsupported lattice")
        end
        n_spin = 2
        dim    = n_sublattices * n_spin

        χ0_zz = zeros(ComplexF64, n_sublattices, n_sublattices)
        χ0_pm = zeros(ComplexF64, n_sublattices, n_sublattices)

        # Assuming generate_kpoints(params.lattice, params.kpoints) is accessible here
        kpoints = generate_kpoints(params.lattice, params.kpoints)

        for k in kpoints
            H_k  = build_hamiltonian(k, params, δm)
            H_kq = build_hamiltonian(k .+ q, params, δm)

            E_k,  U_k  = eigen(Hermitian(H_k))
            E_kq, U_kq = eigen(Hermitian(H_kq))

            for ν in 1:dim, ν′ in 1:dim
                fν  = fermi(E_k[ν], μ, params.β)
                fν′ = fermi(E_kq[ν′], μ, params.β)
                denom = E_kq[ν′] - E_k[ν] + im * η

                spin_ν, sub_ν   = divrem(ν - 1, n_sublattices)
                spin_ν′, sub_ν′ = divrem(ν′ - 1, n_sublattices)

                # χ0^zz: diagonal in spin
                if spin_ν == spin_ν′
                    χ0_zz[sub_ν + 1, sub_ν′ + 1] += 
                        (fν - fν′) * conj(U_k[:,ν][ν]) * U_kq[:,ν′][ν′] / denom
                end

                # χ0^+-: only ↑↓ transitions
                if spin_ν == 0 && spin_ν′ == 1
                    χ0_pm[sub_ν + 1, sub_ν′ + 1] += 
                        (fν - fν′) * conj(U_k[:,ν][ν]) * U_kq[:,ν′][ν′] / denom
                end
            end
        end

        # Normalize by number of kpoints
        χ0_zz ./= length(kpoints)
        χ0_pm ./= length(kpoints)

        max_eig_zz = maximum(real.(eigvals(Hermitian(χ0_zz))))
        max_eig_pm = maximum(real.(eigvals(Hermitian(χ0_pm))))

        return max_eig_zz, max_eig_pm
    end

    
    function compute_chi0_and_Uc(params::ModelParams, μ::Float64, δm::Float64; η=0.01)
        qpoints = generate_kpoints(params.lattice, params.kpoints)

        # Run in parallel across qpoints
        results = pmap(q -> compute_chi0_for_q(q, params, μ, δm, η), qpoints)

        max_eig_zz = maximum(first.(results))
        max_eig_pm = maximum(last.(results))

        Uc_zz = 1.0 / max_eig_zz
        Uc_pm = 1.0 / max_eig_pm

        return Uc_zz, Uc_pm
    end

    function compute_rpa_susceptibility(
        qpath::Vector{Vector{Float64}},   # Each q is a standard Vector
        ωlist::Vector{Float64},
        params::ModelParams,
        μ::Float64,
        δm::Float64,
        U::Float64,
        χtype::Symbol = :zz;
        η::Float64 = 0.01
    )
        results = pmap(q -> compute_rpa_susceptibility_at_qω(q, ωlist, params, μ, δm, U, χtype; η=η), qpath)

        return [(q=q, ω=ω, Imχ=Imχ) for (q, (ω, Imχ)) in zip(qpath, results)]
    end

    
end