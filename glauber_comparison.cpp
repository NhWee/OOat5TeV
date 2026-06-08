#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <iomanip>
#include <string>

// Mass number of Oxygen
const double A_NUCLEUS = 16.0;
const double B_NUCLEUS = 16.0;
const double SIGMA_IN = 7.0; // fm^2 (70 mb for 5.36 TeV)

// Two-component model parameters
const double N_PP = 6.8; 
const double X_HARD = 0.13;

// Structure to hold model parameters
struct WSParams {
    double R;
    double a;
    double w;
};

// Woods-Saxon density distribution (unnormalized)
double ws_unnormalized(double r, const WSParams& p) {
    double numerator = 1.0 + p.w * std::pow(r / p.R, 2);
    double denominator = 1.0 + std::exp((r - p.R) / p.a);
    return numerator / denominator;
}

// Numerical integration of Woods-Saxon to find central density rho_0
double calculate_rho0(const WSParams& p) {
    double sum = 0.0;
    double dr = 0.001;
    double r_max = 20.0;
    
    for (double r = 0.0; r < r_max; r += dr) {
        double r1 = r;
        double r2 = r + dr;
        double val1 = r1 * r1 * ws_unnormalized(r1, p);
        double val2 = r2 * r2 * ws_unnormalized(r2, p);
        sum += 0.5 * (val1 + val2) * dr;
    }
    
    double integral = 4.0 * M_PI * sum;
    return A_NUCLEUS / integral;
}

// Normalized density
double ws_density(double r, const WSParams& p, double rho0) {
    return rho0 * ws_unnormalized(r, p);
}

// Calculate Thickness function T(s) directly
double calculate_T_direct(double s, const WSParams& p, double rho0) {
    double sum = 0.0;
    double dz = 0.01;
    double z_max = 20.0;
    
    for (double z = 0.0; z < z_max; z += dz) {
        double z1 = z;
        double z2 = z + dz;
        double r1 = std::sqrt(s * s + z1 * z1);
        double r2 = std::sqrt(s * s + z2 * z2);
        double val1 = ws_density(r1, p, rho0);
        double val2 = ws_density(r2, p, rho0);
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
    ThicknessTable(const WSParams& p, double rho0) {
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

// Output struct
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
    // 2-parameter Fermi (2pF) parameters
    WSParams p_2pF = {2.608, 0.513, 0.0};
    // 3-parameter Fermi (3pF) parameters (Opar set)
    WSParams p_3pF = {2.608, 0.513, -0.051};
    
    double rho0_2pF = calculate_rho0(p_2pF);
    double rho0_3pF = calculate_rho0(p_3pF);
    
    ThicknessTable T_2pF(p_2pF, rho0_2pF);
    ThicknessTable T_3pF(p_3pF, rho0_3pF);
    
    // Save densities to comparison file
    std::ofstream df("density_comparison.csv");
    df << "r,rho_2pF,rho_3pF,T_2pF,T_3pF\n";
    for (double r = 0.0; r <= 8.0; r += 0.05) {
        df << r << "," 
           << ws_density(r, p_2pF, rho0_2pF) << "," 
           << ws_density(r, p_3pF, rho0_3pF) << ","
           << T_2pF.get_T(r) << ","
           << T_3pF.get_T(r) << "\n";
    }
    df.close();
    
    // Save Glauber calculations
    std::ofstream gf("glauber_comparison.csv");
    gf << "b,Tab_2pF,Tab_3pF,Ncoll_2pF,Ncoll_3pF,Npart_2pF,Npart_3pF,dNch_2pF,dNch_3pF\n";
    
    for (double b = 0.0; b <= 10.0; b += 0.1) {
        GlauberResult r2 = calculate_glauber(b, T_2pF, T_2pF);
        GlauberResult r3 = calculate_glauber(b, T_3pF, T_3pF);
        
        gf << b << ","
           << r2.Tab << "," << r3.Tab << ","
           << r2.Ncoll << "," << r3.Ncoll << ","
           << r2.Npart << "," << r3.Npart << ","
           << r2.dNch_deta << "," << r3.dNch_deta << "\n";
    }
    gf.close();
    
    std::cout << "Glauber comparison run successful. Files saved." << std::endl;
    return 0;
}
