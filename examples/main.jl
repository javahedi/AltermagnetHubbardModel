using AltermagneticHubbardModel
using PyPlot
const plt = PyPlot  # Create an alias for PyPlot


function main()
    # Set up parameters for square lattice altermagnet
    params = ModelParams(
        lattice = ALPHA_T3,  #  SQUARE, HEXATRIANGULAR, ALPHA_T3
        t       = 1.0,
        t_prime = 0.3,
        δ       = 0.8,   # Altermagnetic order parameter
        U       = 4.0,
        λ       = 0.0,  # Rashba SOC strength
        n       = 1.0,  # half-filling
        β       = 10000.0,  # T = 0.0001
        α       = π/12,  # α parameter for T3 lattice
        kpoints = 100,
        mixing  = 0.4,
        tol     = 1e-6
    )

    println("Model Parameters:")
    println(params)
    println("\nRunning SCF calculation...")
    
    # Run self-consistent calculation
    δm_final, m_plus = run_scf(params)
    
    println("\nResults:")
    println("Final altermagnetic order parameter: δm = ", δm_final)
    println("Final net magnetization: m_+ = ", m_plus)

    
    # Plot bands
    fig = plot_band_structure(params, δm_final)
    plt.show()
    
   
    #σ = calculate_conductivity(params, δm_final)
    #println("longitudinal up spin = ", σ.up_longitudinal)
    #println("longitudinal down spin = ", σ.down_longitudinal)
    #println("transverse up spin = ", σ.up_transverse)
    #println("transverse down spin = ", σ.down_transverse)
    #println("Hall up spin = ", σ.up_Hall)
    #println("Hall down spin = ", σ.down_Hall)

  

    #fig = plot_spectral_function(params, δm_final)
    #plt.savefig("spectral_function.pdf")
    #plt.show()
    

    fig = plot_fermi_surface(params, δm_final; nk=100)
    if fig
        plt.show()
    else
        println("Fermi surface plot not generated.")
    end
    
end

# Execute the main function
main()