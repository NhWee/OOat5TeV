#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <iomanip>
#include <string>

// Physics Constants for Oxygen-Oxygen (OO) collisions at 5.36 TeV
const double A_NUCLEUS = 16.0;      // Mass number of Oxygen
const double B_NUCLEUS = 16.0;      // Mass number of Oxygen
const double R_OXYGEN = 2.608;      // Woods-Saxon radius R (fm)
const double A_OXYGEN = 0.513;      // Woods-Saxon skin depth a (fm)
const double SIGMA_IN = 7.0;        // Inelastic nucleon-nucleon cross section (fm^2) equivalent to 70 mb

// Woods-Saxon density distribution (unnormalized)
double woods_saxon_unnormalized(double r) {
    return 1.0 / (1.0 + std::exp((r - R_OXYGEN) / A_OXYGEN));
}

// Numerical integration of Woods-Saxon to find the normalization factor Central Density (rho_0)
double calculate_rho0() {
    double sum = 0.0;
    double dr = 0.001; // Integration step size (fm)
    double r_max = 20.0;
    
    // 3D volume integral: 4 * pi * \int r^2 \rho(r) dr = A
    // We use the trapezoidal rule
    for (double r = 0.0; r < r_max; r += dr) {
        double r1 = r;
        double r2 = r + dr;
        double val1 = r1 * r1 * woods_saxon_unnormalized(r1);
        double val2 = r2 * r2 * woods_saxon_unnormalized(r2);
        sum += 0.5 * (val1 + val2) * dr;
    }
    
    double integral = 4.0 * M_PI * sum;
    return A_NUCLEUS / integral;
}

// Normalized Woods-Saxon density
double woods_saxon(double r, double rho0) {
    return rho0 * woods_saxon_unnormalized(r);
}

// Global precomputed Thickness Function T(s) table
const double S_MAX = 15.0;
const double DS = 0.01;
std::vector<double> T_table;
double rho_central = 0.0;

// Calculate Thickness function T(s) directly by integrating along z-axis
// T(s) = 2 * \int_{0}^{\infty} \rho(\sqrt{s^2 + z^2}) dz
double calculate_T_direct(double s, double rho0) {
    double sum = 0.0;
    double dz = 0.01; // Step size (fm)
    double z_max = 20.0; // Max z (fm)
    
    for (double z = 0.0; z < z_max; z += dz) {
        double z1 = z;
        double z2 = z + dz;
        double r1 = std::sqrt(s * s + z1 * z1);
        double r2 = std::sqrt(s * s + z2 * z2);
        double val1 = woods_saxon(r1, rho0);
        double val2 = woods_saxon(r2, rho0);
        sum += 0.5 * (val1 + val2) * dz;
    }
    return 2.0 * sum;
}

// Initialize thickness table
void initialize_T_table(double rho0) {
    int n_points = static_cast<int>(S_MAX / DS) + 1;
    T_table.resize(n_points);
    for (int i = 0; i < n_points; ++i) {
        double s = i * DS;
        T_table[i] = calculate_T_direct(s, rho0);
    }
}

// Interpolate thickness function T(s) from precomputed table
double T_func(double s) {
    if (s >= S_MAX) return 0.0;
    int idx = static_cast<int>(s / DS);
    double s1 = idx * DS;
    double s2 = (idx + 1) * DS;
    double t1 = T_table[idx];
    double t2 = T_table[idx + 1];
    
    // Linear interpolation
    return t1 + (t2 - t1) * (s - s1) / (s2 - s1);
}

// Calculate Glauber quantities for a given impact parameter b
// Returns a structure containing Tab, Ncoll, and Npart
struct GlauberResult {
    double b;
    double Tab;
    double Ncoll;
    double Npart_A;
    double Npart_B;
    double Npart;
    double dNch_deta;
};

GlauberResult calculate_glauber(double b) {
    double Tab_sum = 0.0;
    double Npart_A_sum = 0.0;
    double Npart_B_sum = 0.0;
    
    // We integrate over a 2D transverse plane (x, y)
    // Range of integration: -10 to 10 fm on both axes
    double xy_limit = 10.0;
    double dxy = 0.05; // Grid spacing (fm)
    
    // Centers of nucleus A and B
    double x_A = -b / 2.0;
    double x_B = b / 2.0;
    
    for (double x = -xy_limit; x < xy_limit; x += dxy) {
        for (double y = -xy_limit; y < xy_limit; y += dxy) {
            // Distances from the centers of A and B
            double s_A = std::sqrt((x - x_A) * (x - x_A) + y * y);
            double s_B = std::sqrt((x - x_B) * (x - x_B) + y * y);
            
            double Ta = T_func(s_A);
            double Tb = T_func(s_B);
            
            // Overlap integrand: T_A(s + b/2) * T_B(s - b/2)
            Tab_sum += Ta * Tb * dxy * dxy;
            
            // Participant probability for a nucleon in A at (x, y)
            // Prob = 1 - (1 - sigma * Tb / B)^B
            double p_interact_A = 1.0 - std::pow(1.0 - (SIGMA_IN * Tb) / B_NUCLEUS, B_NUCLEUS);
            if (p_interact_A < 0.0) p_interact_A = 0.0; // bound check
            Npart_A_sum += Ta * p_interact_A * dxy * dxy;
            
            // Participant probability for a nucleon in B at (x, y)
            // Prob = 1 - (1 - sigma * Ta / A)^A
            double p_interact_B = 1.0 - std::pow(1.0 - (SIGMA_IN * Ta) / A_NUCLEUS, A_NUCLEUS);
            if (p_interact_B < 0.0) p_interact_B = 0.0; // bound check
            Npart_B_sum += Tb * p_interact_B * dxy * dxy;
        }
    }
    
    GlauberResult res;
    res.b = b;
    res.Tab = Tab_sum;
    res.Ncoll = SIGMA_IN * Tab_sum;
    res.Npart_A = Npart_A_sum;
    res.Npart_B = Npart_B_sum;
    res.Npart = Npart_A_sum + Npart_B_sum;
    
    // Two-component model parameters tuned for OO at 5.36 TeV to match dNch/deta(0) = 135
    // n_pp is the multiplicity in pp collisions (~6.8)
    // x_hard is the fraction of hard component (~0.13)
    const double n_pp = 6.8; 
    const double x_hard = 0.13;
    res.dNch_deta = n_pp * ((1.0 - x_hard) * res.Npart / 2.0 + x_hard * res.Ncoll);
    
    return res;
}

int main() {
    std::cout << "====================================================" << std::endl;
    std::cout << " Glauber Model Calculator for Oxygen-Oxygen (OO) " << std::endl;
    std::cout << " Collisions at sqrt(s_NN) = 5.36 TeV" << std::endl;
    std::cout << "====================================================" << std::endl;
    
    // Step 1: Calculate central density normalization
    rho_central = calculate_rho0();
    std::cout << "Central density rho_0 = " << rho_central << " fm^-3" << std::endl;
    
    // Step 2: Initialize T(s) lookup table
    initialize_T_table(rho_central);
    std::cout << "Thickness function T(s) precomputation completed." << std::endl;
    
    // Step 3: Run calculations for b = 0 to 10 fm and output to CSV
    std::string filename = "glauber_results.csv";
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing!" << std::endl;
        return 1;
    }
    
    outfile << "b,Tab,Ncoll,Npart_A,Npart_B,Npart,dNch_deta\n";
    
    std::cout << "\nCalculating Glauber parameters as a function of impact parameter b..." << std::endl;
    std::cout << std::left << std::setw(8) << "b (fm)" 
              << std::setw(12) << "Tab (fm^-2)" 
              << std::setw(10) << "Ncoll" 
              << std::setw(10) << "Npart" 
              << std::setw(12) << "dNch/deta" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    
    for (double b = 0.0; b <= 10.0; b += 0.1) {
        GlauberResult res = calculate_glauber(b);
        outfile << std::fixed << std::setprecision(5)
                << res.b << "," 
                << res.Tab << "," 
                << res.Ncoll << "," 
                << res.Npart_A << "," 
                << res.Npart_B << "," 
                << res.Npart << ","
                << res.dNch_deta << "\n";
                
        // Print every 1.0 fm to stdout
        if (std::abs(std::round(res.b * 10) - res.b * 10) < 1e-5 && 
            static_cast<int>(std::round(res.b * 10)) % 10 == 0) {
            std::cout << std::left << std::fixed << std::setprecision(2)
                      << std::setw(8) << res.b 
                      << std::setprecision(4)
                      << std::setw(12) << res.Tab 
                      << std::setprecision(2)
                      << std::setw(10) << res.Ncoll 
                      << std::setw(10) << res.Npart 
                      << std::setw(12) << res.dNch_deta << std::endl;
        }
    }
    
    outfile.close();
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "Results successfully saved to " << filename << std::endl;
    
    return 0;
}
