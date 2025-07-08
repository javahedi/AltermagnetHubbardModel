using AltermagneticHubbardModel
using PyPlot
const plt = PyPlot  # Create an alias for PyPlot


function main()
    # Set up parameters for square lattice altermagnet
    params = ModelParams(
        lattice = HEXATRIANGULAR,  # Change to SQUARE for square lattice
        t       = 1.0,
        t_prime = 0.0,
        δ       = 0.8, 
        U       = 0.0,
        λ       = 0.0,  # Rashba SOC strength
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
    
   
    σ = calculate_conductivity(params, δm_final)
    println("longitudinal up spin = ", σ.up_longitudinal)
    println("longitudinal down spin = ", σ.down_longitudinal)
    println("transverse up spin = ", σ.up_transverse)
    println("transverse down spin = ", σ.down_transverse)
    println("Hall up spin = ", σ.up_Hall)
    println("Hall down spin = ", σ.down_Hall)

    # Calculate all components of the total charge conductivity tensor
    #σ_charge = calculate_conductivity(params, δm_final)

    #println("Total Charge Conductivity Tensor:")
    #println("σ_xx = ", σ_charge.charge_xx)
    #println("σ_yy = ", σ_charge.charge_yy)
    #println("σ_xy = ", σ_charge.charge_xy)
    #println("σ_yx = ", σ_charge.charge_yx)

    #println("\nDerived Charge Conductivity Components:")
    #println("Longitudinal (σ_xx, σ_yy) = ", σ_charge.charge_longitudinal)
    #println("Transverse Symmetric ( (σ_xy + σ_yx)/2 ) = ", σ_charge.charge_transverse_symmetric)
    #println("Hall Antisymmetric ( (σ_xy - σ_yx)/2 ) = ", σ_charge.charge_Hall_antisymmetric)


    
    #fig = plot_spectral_function(params, δm_final)
    #plt.savefig("spectral_function.pdf")
    #plt.show()
    
    
end

# Execute the main function
main()