#define _USE_MATH_DEFINES
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <iomanip>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Mass number of Oxygen-16
const double A_NUCLEUS = 16.0;
const double B_NUCLEUS = 16.0;
const double SIGMA_IN = 7.0; // fm^2 (70 mb for 5.36 TeV)

// Two-component model parameters
const double N_PP = 6.8; 
const double X_HARD = 0.13;

// Structure to hold profile parameters
struct ProfileParams {
    int type; // 1 for Woods-Saxon (Fermi), 2 for Harmonic Oscillator
    double R_or_alpha; // R for Fermi, alpha for HO
    double a_or_a_ho;  // a for Fermi, a_ho for HO
    double w;          // w parameter for Fermi (0.0 for 2pF)
    std::string name;
};

// Density distribution (unnormalized)
double density_unnormalized(double r, const ProfileParams& p) {
    if (p.type == 1) { // Woods-Saxon (Fermi)
        double numerator = 1.0 + p.w * std::pow(r / p.R_or_alpha, 2);
        double denominator = 1.0 + std::exp((r - p.R_or_alpha) / p.a_or_a_ho);
        return numerator / denominator;
    } else if (p.type == 2) { // Harmonic Oscillator
        double alpha = p.R_or_alpha;
        double a_ho = p.a_or_a_ho;
        return (1.0 + alpha * std::pow(r / a_ho, 2)) * std::exp(-std::pow(r / a_ho, 2));
    }
    return 0.0;
}

// Numerical integration of density to find the normalization factor (rho_0)
// 4 * pi * \int r^2 \rho(r) dr = A_NUCLEUS
double calculate_rho0(const ProfileParams& p) {
    double sum = 0.0;
    double dr = 0.001;
    double r_max = 20.0;
    
    for (double r = 0.0; r < r_max; r += dr) {
        double r1 = r;
        double r2 = r + dr;
        double val1 = r1 * r1 * density_unnormalized(r1, p);
        double val2 = r2 * r2 * density_unnormalized(r2, p);
        sum += 0.5 * (val1 + val2) * dr;
    }
    
    double integral = 4.0 * M_PI * sum;
    return A_NUCLEUS / integral;
}

// Normalized density
double get_density(double r, const ProfileParams& p, double rho0) {
    return rho0 * density_unnormalized(r, p);
}

// Calculate Thickness function T(s) directly
double calculate_T_direct(double s, const ProfileParams& p, double rho0) {
    double sum = 0.0;
    double dz = 0.01;
    double z_max = 20.0;
    
    for (double z = 0.0; z < z_max; z += dz) {
        double z1 = z;
        double z2 = z + dz;
        double r1 = std::sqrt(s * s + z1 * z1);
        double r2 = std::sqrt(s * s + z2 * z2);
        double val1 = get_density(r1, p, rho0);
        double val2 = get_density(r2, p, rho0);
        sum += 0.5 * (val1 + val2) * dz;
    }
    return 2.0 * sum;
}

// Thickness table class for quick interpolation
class ThicknessTable {
private:
    std::vector<double> table;
    double ds;
    double s_max;

public:
    ThicknessTable(const ProfileParams& p, double rho0) {
        ds = 0.01;
        s_max = 15.0;
        int n_points = static_cast<int>(s_max / ds) + 1;
        table.resize(n_points);
        for (int i = 0; i < n_points; ++i) {
            double s = i * ds;
            table[i] = calculate_T_direct(s, p, rho0);
        }
    }

    double get_T(double s) const {
        if (s >= s_max) return 0.0;
        int idx = static_cast<int>(s / ds);
        double s1 = idx * ds;
        double s2 = (idx + 1) * ds;
        double t1 = table[idx];
        double t2 = table[idx + 1];
        return t1 + (t2 - t1) * (s - s1) / (s2 - s1);
    }
};

// Output structure
struct GlauberResult {
    double b;
    double Tab;
    double Ncoll;
    double Npart;
    double dNch_deta;
};

// Calculate overlap integrations
GlauberResult calculate_glauber(double b, const ThicknessTable& T_A, const ThicknessTable& T_B) {
    double Tab_sum = 0.0;
    double Npart_A_sum = 0.0;
    double Npart_B_sum = 0.0;
    
    double xy_limit = 10.0;
    double dxy = 0.05;
    
    double x_A = -b / 2.0;
    double x_B = b / 2.0;
    
    for (double x = -xy_limit; x < xy_limit; x += dxy) {
        for (double y = -xy_limit; y < xy_limit; y += dxy) {
            double s_A = std::sqrt((x - x_A) * (x - x_A) + y * y);
            double s_B = std::sqrt((x - x_B) * (x - x_B) + y * y);
            
            double Ta = T_A.get_T(s_A);
            double Tb = T_B.get_T(s_B);
            
            Tab_sum += Ta * Tb * dxy * dxy;
            
            double p_interact_A = 1.0 - std::pow(1.0 - (SIGMA_IN * Tb) / B_NUCLEUS, B_NUCLEUS);
            if (p_interact_A < 0.0) p_interact_A = 0.0;
            Npart_A_sum += Ta * p_interact_A * dxy * dxy;
            
            double p_interact_B = 1.0 - std::pow(1.0 - (SIGMA_IN * Ta) / A_NUCLEUS, A_NUCLEUS);
            if (p_interact_B < 0.0) p_interact_B = 0.0;
            Npart_B_sum += Tb * p_interact_B * dxy * dxy;
        }
    }
    
    GlauberResult res;
    res.b = b;
    res.Tab = Tab_sum;
    res.Ncoll = SIGMA_IN * Tab_sum;
    res.Npart = Npart_A_sum + Npart_B_sum;
    res.dNch_deta = N_PP * ((1.0 - X_HARD) * res.Npart / 2.0 + X_HARD * res.Ncoll);
    return res;
}

int main() {
    std::cout << "Running Glauber Comparison for 2pF, 3pF, and HO profiles..." << std::endl;
    
    // Model 1: 2pF User Workspace (R=2.82, a=0.546, w=0.0)
    ProfileParams p_user = {1, 2.82, 0.546, 0.0, "2pF_UserWS"};
    // Model 2: 3pF de Vries Standard (R=2.608, a=0.513, w=-0.051)
    ProfileParams p_opar = {1, 2.608, 0.513, -0.051, "3pF_Opar"};
    // Model 3: HO fit to data (alpha=1.506, a_ho=1.819)
    ProfileParams p_oho2 = {2, 1.506, 1.819, 0.0, "HO_Oho2"};
    
    double rho0_user = calculate_rho0(p_user);
    double rho0_opar = calculate_rho0(p_opar);
    double rho0_oho2 = calculate_rho0(p_oho2);
    
    std::cout << "Central Densities (rho0) Matter-normalized to A=16:" << std::endl;
    std::cout << "  2pF UserWS: " << rho0_user << " fm^-3" << std::endl;
    std::cout << "  3pF Opar  : " << rho0_opar << " fm^-3" << std::endl;
    std::cout << "  HO Oho2   : " << rho0_oho2 << " fm^-3" << std::endl;
    
    ThicknessTable T_user(p_user, rho0_user);
    ThicknessTable T_opar(p_opar, rho0_opar);
    ThicknessTable T_oho2(p_oho2, rho0_oho2);
    
    // Save densities to comparison file
    std::ofstream df("density_comparison.csv");
    df << "r,rho_user,rho_opar,rho_oho2,T_user,T_opar,T_oho2\n";
    for (double r = 0.0; r <= 8.0; r += 0.05) {
        df << r << "," 
           << get_density(r, p_user, rho0_user) << "," 
           << get_density(r, p_opar, rho0_opar) << ","
           << get_density(r, p_oho2, rho0_oho2) << ","
           << T_user.get_T(r) << ","
           << T_opar.get_T(r) << ","
           << T_oho2.get_T(r) << "\n";
    }
    df.close();
    std::cout << "Saved density_comparison.csv" << std::endl;
    
    // Save Glauber calculations
    std::ofstream gf("glauber_comparison.csv");
    gf << "b,Tab_user,Tab_opar,Tab_oho2,Ncoll_user,Ncoll_opar,Ncoll_oho2,Npart_user,Npart_opar,Npart_oho2,dNch_user,dNch_opar,dNch_oho2\n";
    
    for (double b = 0.0; b <= 10.0; b += 0.1) {
        GlauberResult r_u = calculate_glauber(b, T_user, T_user);
        GlauberResult r_p = calculate_glauber(b, T_opar, T_opar);
        GlauberResult r_o = calculate_glauber(b, T_oho2, T_oho2);
        
        gf << b << ","
           << r_u.Tab << "," << r_p.Tab << "," << r_o.Tab << ","
           << r_u.Ncoll << "," << r_p.Ncoll << "," << r_o.Ncoll << ","
           << r_u.Npart << "," << r_p.Npart << "," << r_o.Npart << ","
           << r_u.dNch_deta << "," << r_p.dNch_deta << "," << r_o.dNch_deta << "\n";
    }
    gf.close();
    std::cout << "Saved glauber_comparison.csv" << std::endl;
    
    std::cout << "Glauber comparison run successful." << std::endl;
    return 0;
}
