module Utility

    using LinearAlgebra
    using AltermagneticHubbardModel
    using Base.Threads

    export compute_BC, compute_AHE, fermi, compute_AHE_kubo
    export compute_chi_q0, chi_qω
    export chern_tot_and_spin, chern_numbers_FHS, chern_occupied_FHS
    export spin_operator, σx, σy, σz

    const e = 1  # Natural units (e^2/ħ)
    const ħ = 1


    # Define spin Pauli matrices in 2x2 space
    σx = [0.0 1.0; 1.0 0.0]
    σy = [0.0 -1im; 1im 0.0]
    σp = [0.0 1.0; 0.0 0.0]
    σn = [0.0 0.0; 1.0 0.0]
    σz = [1.0 0.0; 0.0 -1.0]
    σ_list = [σx, σy, σz]


    # Build spin operators in 6x6 basis: identity(3) ⊗ σμ
    function spin_operator(σμ, dim)
        Idim = Matrix(I, dim, dim)
        return kron(Idim, σμ) / 2  # spin-½ operators
    end


    """
        fermi(ϵ, μ, β)

    Fermi-Dirac distribution function.
    """
    function fermi(ϵ::Float64, μ::Float64, β::Float64)
        1.0 / (exp(β * (ϵ - μ)) + 1.0)
    end

    """
        compute_BC(k::Tuple{Float64,Float64}, params::ModelParams, δm::Float64; η=0.01)

    Compute the Berry curvature Ωₙ(𝐤) for all bands at a given k-point.
    - η: Broadening parameter (default: 0.01)
    """
    function compute_BC(k::Tuple{Float64,Float64}, params::ModelParams, δm::Float64; η=0.01)
        H = build_hamiltonian(k, params, δm)
        ϵ, ψ = eigen(Hermitian(H))
        
        Δk = 1e-5  # Finite difference step
        kx_plus  = (k[1]+Δk, k[2])
        kx_minus = (k[1]-Δk, k[2])
        ky_plus  = (k[1], k[2]+Δk)
        ky_minus = (k[1], k[2]-Δk)
        
        # Velocity operators via finite differences
        ∂xH = (build_hamiltonian(kx_plus, params, δm) - build_hamiltonian(kx_minus, params, δm)) / (2Δk)
        ∂yH = (build_hamiltonian(ky_plus, params, δm) - build_hamiltonian(ky_minus, params, δm)) / (2Δk)
        
        Ω = zeros(length(ϵ))
        for n in eachindex(ϵ), m in eachindex(ϵ)
            if m ≠ n
                vx_mn = ψ[:, m]' * ∂xH * ψ[:, n]
                vy_mn = ψ[:, n]' * ∂yH * ψ[:, m]
                term = vx_mn * vy_mn / (ϵ[m] - ϵ[n] + im*η)^2
                Ω[n] -= 2 * imag(term)
                
                # Numerical stability check
                if abs(imag(term)) > 1e-8 * abs(real(term))
                    @warn "Large imaginary component in Berry curvature at k=$k, bands ($n,$m)"
                end
            end
        end
        return real.(Ω)  # Ensure purely real output
    end


    function compute_BC_FHS(k::Tuple{Float64,Float64}, params::ModelParams, δm::Float64)
        Δk = 2π/params.kpoints  # Match your k-grid spacing
        kx, ky = k
        
        # Define 4 k-points forming a small plaquette
        k_points = [
            (kx, ky),          # k1
            (kx + Δk, ky),     # k2
            (kx + Δk, ky + Δk),# k3
            (kx, ky + Δk)      # k4
        ]
        
        # Get eigenstates at each k-point
        U = Vector{Matrix{ComplexF64}}(undef, 4)
        for i in 1:4
            H = build_hamiltonian(k_points[i], params, δm)
            _, ψ = eigen(Hermitian(H))
            U[i] = ψ
        end
        
        n_bands = size(U[1], 2)
        Ω = zeros(n_bands)
        
        for n in 1:n_bands
            # Calculate U(1) link variables
            links = ComplexF64[
                U[1][:,n]' * U[2][:,n],  # U12
                U[2][:,n]' * U[3][:,n],  # U23
                U[3][:,n]' * U[4][:,n],  # U34
                U[4][:,n]' * U[1][:,n]   # U41
            ]
            
            # Handle numerical zeros (avoid NaN)
            links = map(x -> isapprox(abs(x), 0, atol=1e-10) ? complex(1.0) : x/abs(x), links)
            
            # Berry curvature for this band
            F = prod(links)
            Ω[n] = angle(F)/(Δk^2)
        end

        return real.(Ω)
    end



    """
        compute_AHE_kubo(params::ModelParams, δm::Float64, μ::Float64; η=0.01)
    """
    function compute_AHE_kubo(params::ModelParams, δm::Float64, μ::Float64; η=0.01)
        kpoints = generate_kpoints(params.lattice, params.kpoints)
        σ_xy = Threads.Atomic{Float64}(0.0)
        
        @threads for k in kpointsv
            H = build_hamiltonian(k, params, δm)
            ϵ, ψ = eigen(Hermitian(H))
            vx = (build_hamiltonian(k .+ (1e-5,0), params, δm) - build_hamiltonian(k .- (1e-5,0), params, δm)) / (2e-5)
            vy = (build_hamiltonian(k .+ (0,1e-5), params, δm) - build_hamiltonian(k .- (0,1e-5), params, δm)) / (2e-5)
            for n in eachindex(ϵ), m in eachindex(ϵ)
                if m ≠ n
                    f_n = fermi(ϵ[n], μ, params.β)
                    vx_mn = ψ[:,m]' * vx * ψ[:,n]
                    vy_nm = ψ[:,n]' * vy * ψ[:,m]
                    term = imag(vx_mn * vy_nm) * (f_n - fermi(ϵ[m], μ, params.β)) / 
                        ((ϵ[m] - ϵ[n])^2 + η^2)
                    Threads.atomic_add!(σ_xy, term)
                end
            end
        end
        return -e^2/ħ * σ_xy.value / length(kpoints)
    end



    """
        compute_AHE(params::ModelParams, δm::Float64, μ::Float64; β=100.0, η=0.01, nk=100, bc_method=:FHS)

    Compute the anomalous Hall conductivity σₓᵧ.
    - `params`: Model parameters (lattice, kpoints, etc.)
    - `δm`: Altermagnetic order parameter
    - `μ`: Chemical potential
    - `β`: Inverse temperature (default: 100.0, in inverse energy units)
    - `η`: Broadening for Berry curvature (default: 0.01, in energy units)
    - `nk`: Number of k-points per dimension (default: 100)
    - `bc_method`: Berry curvature method (:FHS or :velocity, default: :FHS)
    """
    function compute_AHE(params::ModelParams, δm::Float64, μ::Float64; 
                        β=100.0, η=0.01, bc_method=:FHS)

        @assert β > 0 "Inverse temperature β must be positive"

        σ_xy = Threads.Atomic{Float64}(0.0)
        kpoints = generate_kpoints(params.lattice, params.kpoints)
        
        @threads for k in kpoints
            H = build_hamiltonian(k, params, δm)
            ϵ, _ = eigen(Hermitian(H))
            Ω = bc_method == :FHS ? compute_BC_FHS(k, params, δm) : compute_BC(k, params, δm; η=η)
            
            for n in eachindex(ϵ)
                f = fermi(ϵ[n], μ, β)
                Threads.atomic_add!(σ_xy, real(f * Ω[n]))
            end
        end
        
        return -e^2 / ħ * σ_xy.value / length(kpoints)  # Units: e²/ħ
    end



                                
    function chi_qω(q::Tuple{Float64,Float64}, ω::Float64 ,params::ModelParams, μ::Float64, δm::Float64; η=1e-2)

        if params.lattice == ALPHA_T3
            n_sublattices = 3
        elseif params.lattice == SQUARE || params.lattice == HONEYCOMB || params.lattice == KMmodel
            n_sublattices = 2
        else
            error("Unsupported lattice")
        end
        n_spin = 2
        dim = n_sublattices * n_spin

        χ0_zz = zeros(ComplexF64, n_sublattices, n_sublattices)
        χ0_pm = zeros(ComplexF64, n_sublattices, n_sublattices)
        χ0_mp = zeros(ComplexF64, n_sublattices, n_sublattices)

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
                denom = ω + E_kq[ν′] - E_k[ν] + im * η

                spin_ν, sub_ν   = divrem(ν - 1, n_sublattices)
                spin_ν′, sub_ν′ = divrem(ν′ - 1, n_sublattices)
                contrib = (fν - fν′) * conj(U_k[:,ν][ν]) * U_kq[:,ν′][ν′] / denom

                # χ0^zz: diagonal in spin
                if spin_ν == spin_ν′
                    χ0_zz[sub_ν + 1, sub_ν′ + 1] += contrib
                end

                # χ0^+-: only ↑↓ transitions
                if spin_ν == 0 && spin_ν′ == 1
                    χ0_pm[sub_ν + 1, sub_ν′ + 1] += contrib
                end

               #χ0^-+: only ↓↑ transitions
               if spin_ν == 1 && spin_ν′ == 0 
                    χ0_mp[sub_ν + 1, sub_ν′ + 1] += contrib
                end

            end
        end

        # Normalize by number of kpoints
        χ0_zz ./= length(kpoints)
        χ0_pm ./= length(kpoints)
        χ0_mp ./= length(kpoints)

        return χ0_zz , χ0_pm, χ0_mp
    end



    function compute_chi_q0(params, μ::Float64, δm::Float64; η=1e-2)
        qpoints = generate_kpoints(params.lattice, params.kpoints)
        chi_zz = zeros(Float64, length(qpoints))
        chi_pm = zeros(Float64, length(qpoints))
        chi_mp = zeros(Float64, length(qpoints))

        @threads for i in eachindex(qpoints)
            q = qpoints[i]
            χ0_zz , χ0_pm, χ0_mp = chi_qω(q, 0.0,params, μ, δm; η=η)
            chi_zz[i] = maximum(real.(eigvals(Hermitian(χ0_zz))))
            chi_pm[i] = maximum(real.(eigvals(Hermitian(χ0_pm))))
            chi_mp[i] = maximum(real.(eigvals(Hermitian(χ0_mp))))

        end

        return qpoints, chi_zz, chi_pm, chi_mp
    end


    ###############################
    # Fukui–Hatsugai–Suzuki (FHS) #
    ###############################

    # Wrap angle to (-π, π]
    @inline function angle_pi(z::Complex)
        θ = atan(imag(z), real(z))
        return θ == -π ? π : θ
    end

    # Go to next index with periodic boundary conditions
    @inline nextidx(i, N) = (i % N) + 1



    # Align H's eigenvectors with Sz inside each degenerate block
    function eigen_with_Sz_alignment(H::AbstractMatrix, Sz::AbstractMatrix; tol=1e-9)
        E, U = eigen(Hermitian(H))               # U columns = eigenvectors
        E = collect(E)                           # Vector
        U = Array(U)                             # Matrix

        # Find clusters of (nearly) equal eigenvalues
        N = length(E)
        used = falses(N)
        for i in 1:N
            used[i] && continue
            # Find indices j with |E[j]-E[i]| < tol  (degenerate block)
            idx = [j for j in i:N if abs(E[j] - E[i]) < tol]
            for j in idx; used[j] = true; end
            if length(idx) > 1
                # Restrict Sz to this block and diagonalize there
                Ublk = U[:, idx]                         # (dim × nblk)
                Sblk = Hermitian(Ublk' * Sz * Ublk)      # (nblk × nblk)
                λs, W = eigen(Sblk)                      # W unitary in the block
                U[:, idx] = Ublk * W                     # rotate to Sz eigenvectors
                # (Optional) sort by descending <Sz> so the band order is stable
                svals = real.(diag(W' * Sblk * W))
                perm = sortperm(svals, rev=true)
                U[:, idx] = U[:, idx][:, perm]
                E[idx] = E[idx][perm]
            end
        end
        return E, U
    end

    # ---------- 1) Abelian FHS: per-band Chern numbers ----------
    """
        chern_numbers_FHS(params::ModelParams, δm; nk=params.kpoints)

    Per-band Chern numbers on an nk×nk periodic grid returned by `generate_kpoints`.
    Assumes `generate_kpoints` iterates i=0..nk-1 (outer), j=0..nk-1 (inner), as shown.

    Returns: Vector{Float64} (≈ integers).
    """
    function chern_numbers_FHS(params::ModelParams, δm::Float64,  Sz::AbstractMatrix; nk::Int=params.kpoints)
        # 1) Fetch k-points and build eigenvectors on an nk×nk array
        klist = generate_kpoints(params.lattice, nk)
        # probe size
        H0 = build_hamiltonian(klist[1], params, δm)
        nb = size(H0, 1)
        U = Array{Matrix{ComplexF64}}(undef, nk, nk)   # eigenvectors (columns are bands)

        Threads.@threads for i in 1:nk
            for j in 1:nk
                idx = (i-1)*nk + j
                k = klist[idx]
                #_, ψ = eigen(Hermitian(build_hamiltonian(k, params, δm)))
                _, ψ = eigen_with_Sz_alignment(build_hamiltonian(k, params, δm), Sz)
                U[i, j] = ψ
            end
        end

        # 2) Accumulate plaquette phases for each band
        C = zeros(Float64, nb)
        for i in 1:nk, j in 1:nk
            i1 = nextidx(i, nk); j1 = nextidx(j, nk)
            ψ    = U[i,  j]
            ψ_u  = U[i1, j]
            ψ_v  = U[i,  j1]
            ψ_uv = U[i1, j1]

            @inbounds for n in 1:nb
                Uu  = dot(conj.(ψ[:,n]),   ψ_u[:,n]);  Uu  = Uu==0 ? 1+0im : Uu/abs(Uu)
                Uv  = dot(conj.(ψ[:,n]),   ψ_v[:,n]);  Uv  = Uv==0 ? 1+0im : Uv/abs(Uv)
                Uu2 = dot(conj.(ψ_v[:,n]), ψ_uv[:,n]); Uu2 = Uu2==0 ? 1+0im : Uu2/abs(Uu2)
                Uv2 = dot(conj.(ψ_u[:,n]), ψ_uv[:,n]); Uv2 = Uv2==0 ? 1+0im : Uv2/abs(Uv2)

                F = Uu * Uv2 * conj(Uu2) * conj(Uv)   # Wilson loop around the plaquette
                C[n] += angle_pi(F)
            end
        end
        return C ./ (2π)
    end


    # ---------- 2) Non-Abelian FHS: Chern of occupied subspace ----------
    """
        chern_occupied_FHS(params, δm, μ; nk=params.kpoints, β=200.0)

    Chern number of the occupied subspace via non-Abelian FHS (determinant of overlap
    matrices) using the nk×nk grid from `generate_kpoints`.
    """
    function chern_occupied_FHS(params::ModelParams, δm::Float64, μ::Float64;
                                    nk::Int=params.kpoints, β::Float64=params.β)
        klist = generate_kpoints(params.lattice, nk)

        # collect orthonormal bases of occupied subspace at each grid point
        Uocc = Array{Matrix{ComplexF64}}(undef, nk, nk)
        nocc_ref = Ref{Int}()
        Threads.@threads for i in 1:nk
            for j in 1:nk
                k = klist[(i-1)*nk + j]
                E, ψ = eigen(Hermitian(build_hamiltonian(k, params, δm)))
                f = 1.0 ./ (exp.(β .* (E .- μ)) .+ 1.0)
                occ = findall(x -> x > 0.5, f)  # T≈0 selection
                if i==1 && j==1
                    nocc_ref[] = length(occ)
                    @assert nocc_ref[] > 0 "No occupied bands at the first k-point."
                else
                    @assert length(occ) == nocc_ref[] "Occupied count changes; ensure μ in a global gap."
                end
                # Orthonormal basis of the occupied subspace (columns)
                Uocc[i, j] = ψ[:, occ]
            end
        end

        C = 0.0
        for i in 1:nk, j in 1:nk
            i1 = nextidx(i, nk); j1 = nextidx(j, nk)
            U0  = Uocc[i,  j]
            Uu  = Uocc[i1, j]
            Uv  = Uocc[i,  j1]
            Uuv = Uocc[i1, j1]

            M_u  = U0' * Uu
            M_v  = U0' * Uv
            M_u2 = Uv' * Uuv
            M_v2 = Uu' * Uuv

            ulink  = det(M_u);  ulink  = ulink==0 ? 1+0im : ulink/abs(ulink)
            vlink  = det(M_v);  vlink  = vlink==0 ? 1+0im : vlink/abs(vlink)
            ulink2 = det(M_u2); ulink2 = ulink2==0 ? 1+0im : ulink2/abs(ulink2)
            vlink2 = det(M_v2); vlink2 = vlink2==0 ? 1+0im : vlink2/abs(vlink2)

            F = ulink * vlink2 * conj(ulink2) * conj(vlink)
            C += angle_pi(F)
        end

        return C / (2π)
    end


    # ---------- 3) Convenience wrappers ----------

    """
        chern_tot_and_spin(params, δm, μ, Sz; nk=params.kpoints, β=200.0, tol=1e-10)

    Spin-resolved Chern numbers from projected occupied subspaces using Sz.
    Returns (C_up, C_dn, C_tot, C_spin).

    - `Sz` must be the spin-z operator in the same single-particle basis as H(k).
    """
    function chern_tot_and_spin(params::ModelParams, δm::Float64, μ::Float64, Sz::AbstractMatrix;
                                    nk::Int=params.kpoints, β::Float64=params.β, tol::Float64=1e-10)
        klist = generate_kpoints(params.lattice, nk)
        I_ = Matrix(I, size(Sz,1), size(Sz,2))
        Pup = 0.5*(I_ + 2Sz)
        Pdn = 0.5*(I_ - 2Sz)

        # Helper: projected occupied bundle at all k (orthonormalized)
        function projected_bundle(P)
            Uproj = Array{Matrix{ComplexF64}}(undef, nk, nk)
            nref = Ref{Int}()
            for i in 1:nk, j in 1:nk
                k = klist[(i-1)*nk + j]
                E, ψ = eigen(Hermitian(build_hamiltonian(k, params, δm)))
                f = 1.0 ./ (exp.(β .* (E .- μ)) .+ 1.0)
                occ = findall(x -> x > 0.5, f)
                ψocc = ψ[:, occ]
                Ψ = P * ψocc
                # QR to orthonormalize; drop nearly-null columns
                Q, R = qr(Ψ)
                keep = [ii for ii in 1:size(R,1) if abs(R[ii,ii]) > tol]
                Uproj[i, j] = Q[:, keep]
                if i==1 && j==1
                    nref[] = size(Uproj[i, j], 2)
                else
                    @assert size(Uproj[i, j], 2) == nref[] "Projected occupied dimension varies; check spin purity & gap."
                end
            end
            return Uproj
        end

        Uup = projected_bundle(Pup)
        Udn = projected_bundle(Pdn)

        # Non-Abelian FHS sum for a precomputed bundle U[i,j]
        function bundle_chern(U)
            C = 0.0
            for i in 1:nk, j in 1:nk
                i1 = nextidx(i, nk); j1 = nextidx(j, nk)
                U0  = U[i,  j]
                Uu  = U[i1, j]
                Uv  = U[i,  j1]
                Uuv = U[i1, j1]
                M_u  = U0' * Uu
                M_v  = U0' * Uv
                M_u2 = Uv' * Uuv
                M_v2 = Uu' * Uuv
                ulink  = det(M_u);  ulink  = ulink==0 ? 1+0im : ulink/abs(ulink)
                vlink  = det(M_v);  vlink  = vlink==0 ? 1+0im : vlink/abs(vlink)
                ulink2 = det(M_u2); ulink2 = ulink2==0 ? 1+0im : ulink2/abs(ulink2)
                vlink2 = det(M_v2); vlink2 = vlink2==0 ? 1+0im : vlink2/abs(vlink2)
                F = ulink * vlink2 * conj(ulink2) * conj(vlink)
                C += angle_pi(F)
            end
            return C/(2π)
        end

        Cup = bundle_chern(Uup)
        Cdn = bundle_chern(Udn)
        return (C_up=Cup, C_dn=Cdn, C_tot=Cup+Cdn, C_spin=0.5*(Cup-Cdn))
    end
 


end