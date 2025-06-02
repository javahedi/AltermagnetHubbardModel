using AltermagneticHubbardModel
using PyPlot
const plt = PyPlot  # Create an alias for PyPlot


function main()
    # Set up parameters for square lattice altermagnet
    params = ModelParams(
        lattice = SQUARE,  # Change to SQUARE for square lattice
        t       = 1.0,
        t_prime = 0.3,
        δ       = 0.8,
        U       = 3.5,
        n       = 1.0,  # half-filling
        β       = 10000.0,  # T = 0.0001
        kpoints = 100,
        mixing  = 0.4,
        tol     = 1e-6
    )

    println("Model Parameters:")
    println(params)
    println("\nRunning SCF calculation...")
    
    # Run self-consistent calculation
    δm_final = run_scf(params)
    
    println("\nResults:")
    println("Final altermagnetic order parameter: δm = ", δm_final)
    
    
    # Plot bands
    fig = plot_band_structure(params, δm_final)
    plt.show()
    
   
    # Save then show
    #plt.savefig("band_structure_U$(params.U).pdf",
    #           bbox_inches="tight",
    #           dpi=300,
    #           facecolor='w',
    #           edgecolor='w')
    # 
    #println("Successfully saved band_structure_U$(params.U).pdf")
    #fig = plot_fermi_surface_comparison(params, δm_final)
    #plt.savefig("fermi_comparison.pdf", bbox_inches="tight", dpi=300)
    #plt.show()

    
    # Calculate all components
    σ = calculate_conductivity(params, δm_final)

    println("Spin-up:")
    println("σ_xx = ", σ.up_longitudinal[1], " σ_yy = ", σ.up_longitudinal[2])
    println("Transverse = ", σ.up_transverse)
    println("Hall up = ", σ.up_Hall)

    println("\nSpin-down:")
    println("σ_xx = ", σ.dn_longitudinal[1], " σ_yy = ", σ.dn_longitudinal[2])
    println("Transverse = ", σ.dn_transverse)
    println("Hall dn = ", σ.dn_Hall)
    println("Spin Hall = ", σ.spin_Hall)
    σ_spin_Hall = (σ.up_Hall - σ.dn_Hall) / 2
    println("Spin Hall conductivity = ", σ_spin_Hall)

    
    fig = plot_spectral_function(params, δm_final)
    #plt.savefig("spectral_function.pdf")
    plt.show()
    
    
end

# Execute the main function
main()