using AltermagneticHubbardModel
using PyPlot
const plt = PyPlot  # Alias for clarity




"""
    main()

Run the self-consistent loop, plot band structure, and visualize χ₀(q).
"""
function main()
    # Define model parameters
    params = ModelParams(
        lattice = ALPHA_T3,         # Options: SQUARE, HEXATRIANGULAR, ALPHA_T3
        t       = 1.0,
        t_prime = 0.3,
        δ       = 0.8,            # Altermagnetic staggered order
        U       = 3.8,
        λ       = 0.1,            # Rashba SOC
        n       = 1.0,            # Half-filling
        β       = 10000.0,        # Low temperature (T ≈ 0.0001)
        α       = π/4,            # Only relevant for T3 lattice
        kpoints = 50,
        mixing  = 0.4,
        tol     = 1e-6
    )

    println("🧮 Model Parameters:\n", params)

    println("\n⚙️ Running SCF calculation...")
    δm_final, m_plus, μ_final = run_scf(params)

    println("\n✅ Results:")
    println("• Final altermagnetic order δm  = ", δm_final)
    println("• Net magnetization      m₊     = ", m_plus)
    println("• Chemical potential     μ      = ", μ_final)


     #    For example, using your helper:
    dim = (params.lattice == ALPHA_T3) ? 3 : 2
    Sz = spin_operator(σz, dim)  # 2*dim × 2*dim

    # 1) Per-band Chern numbers (all bands)
    chern_bands = chern_numbers_FHS(params, δm_final, Sz; nk=400)
    println("Per-band Chern numbers = ", round.(chern_bands; digits=6))

 

    # 2) Occupied-subspace Chern (insulating case)
    Cocc = chern_occupied_FHS(params, δm_final, μ_final; nk=400)
    println("Chern (occupied subspace) = ", round(Cocc; digits=6))

    # 3) (Optional) Spin-resolved totals if you have Sz in the same basis
    C = chern_tot_and_spin(params, δm_final, μ_final, Sz; nk=400)
    @show C.C_tot C.C_spin C.C_up C.C_dn


    # Band structure
    _,_= plot_band_structure(params, δm_final, μ_final; npoints=100, 
                        savepath="band_structure", showlegend=true)
    #plt.show()

    _ = plot_color_coded_band_structure(params, δm_final, μ_final; 
                                    savepath="dos_colored_band_structure")

    _ = plot_fermi_surface(params, δm_final, μ_final; 
                        nk=100, savepath="fermi_surface")

    _ = plot_spectral_function2(params, δm_final, μ_final; 
                            savepath="spectral_function")
    

    #_ = plot_fermi_surface_imshow(params, δm_final, μ_final; 
    #                    nk=100, savepath="fermi_surface_imshow")
    # Compute and plot susceptibility χ₀(q)
    #qpoints, chi_values = compute_chi0_leading_eigenvalue_parallel(params, μ_final, 0.0; η=0.01)
    #plot_chi_0_fbz(qpoints, chi_values)
end


# Run the main routine
main()
