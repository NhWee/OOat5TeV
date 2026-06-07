#include <fstream>
#include <string>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <cstring>
#include <thread>
#include <direct.h>
#include <chrono>

#define PI 3.1415926535897932385

//===================================================================================================
void Nk1(); void Nk2(); void Nk3(); void Nk4(); void Nk5(); void Nk6(); void Nk7();
void Nk8(); void Nk9(); void Nk10(); void Nk11();
//===================================================================================================
int A=16; int B=16; double Ra=2.82; double Rb=2.82; double psmx=2.*PI; int n=100;

double sigjp=0.14; double kappa=25.; double t0=0.43; double zeta=0.04;
double NchCons=(2./3.)*kappa; double IXsecNN=6.80; //fm^2
//===================================================================================================
double Thpcd(double ss) // radius -> nor
{
    double B_local=0.13*16.;
    return 1./(2.*PI*B_local)*exp(-pow(ss,2)/(2*B_local));
}

double Th(double R, double b)
{
    double Thick=0.;
    if(R>b) Thick=sqrt(R*R-b*b)*3./(2.*PI*pow(R,3));
    else Thick=0.;
    return Thick;
}

double Pjetnor(double b)
{
    double Jx=0.; double Jy=0.; double dJ=2.*Ra/n; double sum=0.;
    double JA=0.; double JB=0.;
    for(int i=0; i<n; i++)
    {
        Jx=-Ra+i*dJ;
        for(int j=0; j<n; j++)
        {
            Jy=-Ra+j*dJ;
            JA=sqrt(pow(Jx,2)+pow(Jy,2)); JB=sqrt(pow(Jx-b,2)+pow(Jy,2));
            if(JA<Ra && JB<Rb) sum+=A*B*Thpcd(JA)*Thpcd(JB)*dJ*dJ;
            else continue;
        }
    }
    return sum;
}

//===================================================================================================
double Nk(double b, double Jx, double Jy, double ps)
{
    double t=0.; double tR=0.; double dl=0.; double l=0.; double sumL=0.;
    double CAx=0.; double Cy=0.; double CA=0.; double CBx=0.; double CB=0.;
    double IXBn=0.;

    for(int i=0; i<n; i++)
    {
        dl=1.6/n; l=t0+i*dl; t=l;
        CAx=Jx+l*cos(ps); CBx=CAx-b; Cy=Jy+l*sin(ps);
        CA=sqrt(pow(CAx,2)+pow(Cy,2)); CB=sqrt(pow(CBx,2)+pow(Cy,2));
        IXBn=1.-pow(1.-Thpcd(CB)*IXsecNN,B);
        if(CA<Ra && CB<Rb) sumL+=sigjp*kappa/(2.*t)*(A*Thpcd(CA)*IXBn+B*Thpcd(CB)*Thpcd(CA)*IXsecNN)*dl;
        else continue;
    }
    return sumL;
}

//===================================================================================================
double NptnOFNtrg(double b, double ps)
{
    double Jx=0.; double Jy=0.; double dJ=2.*Ra/n;
    double sumNptn=0.; double sumNtrg=0.; double Pnor=Pjetnor(b);
    double Nk_l=0.; double Pjet=0.; double EXP=0.; double JA=0.; double JB=0.;
    for(int i=0; i<n; i++)
    {
        Jx=-Ra+i*dJ;
        for(int j=0; j<n; j++)
        {
            Jy=-Ra+j*dJ;

            JA=sqrt(pow(Jx,2)+pow(Jy,2)); JB=sqrt(pow(Jx-b,2)+pow(Jy,2));
            if(JA<Ra && JB<Rb) Pjet=A*B*Thpcd(JA)*Thpcd(JB)/Pnor;
            else continue;
            Nk_l=Nk(b,Jx,Jy,ps); EXP=exp(-zeta*Nk_l);
            sumNptn+=Nk_l*EXP*Pjet*dJ*dJ;
            sumNtrg+=EXP*Pjet*dJ*dJ;
        }
    }
    double NptnOFNtrg_val=0.;
    if(sumNptn>0.0001 && sumNtrg>0.0001) NptnOFNtrg_val=sumNptn/sumNtrg;
    else NptnOFNtrg_val=0.;
    return NptnOFNtrg_val;
}

//===================================================================================================
double Nkaver(double b)
{
    double dps=psmx/n; double ps=0.; double sumps=0.;
    for(int i=0; i<n; i++) {ps=i*dps; sumps+=NptnOFNtrg(b,ps)*dps;}
    double Nkaver_val=sumps/psmx;
    return Nkaver_val;
}

//===================================================================================================
double Nch(double b)
{
    double Cx=0.; double Cy=0.; double dC=2.*Ra/n;
    double CA=0.; double CB=0.; double sumNch=0.; double IXBn=0.;
    for(int i=0; i<n; i++)
    {
        Cx=-Ra+i*dC;
        for(int j=0; j<n; j++)
        {
            Cy=-Ra+j*dC;

            CA=sqrt(pow(Cx,2)+pow(Cy,2)); CB=sqrt(pow(Cx-b,2)+pow(Cy,2));
            IXBn=1.-pow(1.-Thpcd(CB)*IXsecNN,B);
            if(CA<Ra && CB<Rb) sumNch+=NchCons*(A*Thpcd(CA)*IXBn+B*Thpcd(CB)*Thpcd(CA)*IXsecNN)*dC*dC;
            else continue;
        }
    }
    return sumNch;
}

//===================================================================================================
int main()
{
    const char* new_directory = "C:\\Users\\Administrator\\Desktop\\MKM\\Workspace\\OO_5.36TeV\\Dataset1";
    if (_chdir(new_directory) != 0) {
        std::cerr << "Error changing directory to " << new_directory << std::endl;
    } else {
        std::cout << "Successfully changed directory to " << new_directory << std::endl;
    }
    std::chrono::steady_clock sc;
    auto start=sc.now();

    std::thread t1(Nk1); std::thread t2(Nk2); std::thread t3(Nk3); std::thread t4(Nk4); std::thread t5(Nk5);
    std::thread t6(Nk6); std::thread t7(Nk7); std::thread t8(Nk8); std::thread t9(Nk9); std::thread t10(Nk10); std::thread t11(Nk11);
 
    t1.join(); t2.join(); t3.join(); t4.join(); t5.join(); t6.join(); t7.join(); t8.join(); t9.join(); t10.join(); t11.join();
 
    std::string filename1="cmd /c type Nk_1.csv Nk_2.csv Nk_3.csv Nk_4.csv Nk_5.csv Nk_6.csv Nk_7.csv Nk_8.csv Nk_9.csv Nk_10.csv Nk_11.csv >IX_b_Nch_Nk.csv";
    system(filename1.c_str());

    std::string filename2="cmd /c del Nk_*.csv";
    system(filename2.c_str());

    std::string filename3="python ../Nk_plot.py";
    system(filename3.c_str());

    auto end=sc.now();
    auto time_span=static_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Nch and Nk calculation took: " << time_span.count() << " seconds" << std::endl;

    return 0;
}
//===================================================================================================
void Nk1()
{
    double b0=0.; double b=0.; double db=0.002;
    std::fstream fs; fs.open("Nk_1.csv", std::ios::out);
    for(int i=0; i<250; i++)
    {
        b=b0+i*db;
        fs << b << "," << Nch(b) << "," << Nkaver(b) << std::endl;
    }
    fs.close();
}
void Nk2()
{
    double b0=0.50; double b=0.; double db=0.002;
    std::fstream fs; fs.open("Nk_2.csv", std::ios::out);
    for(int i=0; i<250; i++)
    {
        b=b0+i*db;
        fs << b << "," << Nch(b) << "," << Nkaver(b) << std::endl;
    }
    fs.close();
}
void Nk3()
{
    double b0=1.00; double b=0.; double db=0.002;
    std::fstream fs; fs.open("Nk_3.csv", std::ios::out);
    for(int i=0; i<250; i++)
    {
        b=b0+i*db;
        fs << b << "," << Nch(b) << "," << Nkaver(b) << std::endl;
    }
    fs.close();
}
void Nk4()
{
    double b0=1.50; double b=0.; double db=0.002;
    std::fstream fs; fs.open("Nk_4.csv", std::ios::out);
    for(int i=0; i<250; i++)
    {
        b=b0+i*db;
        fs << b << "," << Nch(b) << "," << Nkaver(b) << std::endl;
    }
    fs.close();
}
void Nk5()
{
    double b0=2.00; double b=0.; double db=0.002;
    std::fstream fs; fs.open("Nk_5.csv", std::ios::out);
    for(int i=0; i<250; i++)
    {
        b=b0+i*db;
        fs << b << "," << Nch(b) << "," << Nkaver(b) << std::endl;
    }
    fs.close();
}
void Nk6()
{
    double b0=2.50; double b=0.; double db=0.002;
    std::fstream fs; fs.open("Nk_6.csv", std::ios::out);
    for(int i=0; i<250; i++)
    {
        b=b0+i*db;
        fs << b << "," << Nch(b) << "," << Nkaver(b) << std::endl;
    }
    fs.close();
}
void Nk7()
{
    double b0=3.00; double b=0.; double db=0.002;
    std::fstream fs; fs.open("Nk_7.csv", std::ios::out);
    for(int i=0; i<250; i++)
    {
        b=b0+i*db;
        fs << b << "," << Nch(b) << "," << Nkaver(b) << std::endl;
    }
    fs.close();
}
void Nk8()
{
    double b0=3.50; double b=0.; double db=0.002;
    std::fstream fs; fs.open("Nk_8.csv", std::ios::out);
    for(int i=0; i<250; i++)
    {
        b=b0+i*db;
        fs << b << "," << Nch(b) << "," << Nkaver(b) << std::endl;
    }
    fs.close();
}
void Nk9()
{
    double b0=4.00; double b=0.; double db=0.002;
    std::fstream fs; fs.open("Nk_9.csv", std::ios::out);
    for(int i=0; i<250; i++)
    {
        b=b0+i*db;
        fs << b << "," << Nch(b) << "," << Nkaver(b) << std::endl;
    }
    fs.close();
}
void Nk10()
{
    double b0=4.50; double b=0.; double db=0.002;
    std::fstream fs; fs.open("Nk_10.csv", std::ios::out);
    for(int i=0; i<250; i++)
    {
        b=b0+i*db;
        fs << b << "," << Nch(b) << "," << Nkaver(b) << std::endl;
    }
    fs.close();
}
void Nk11()
{
    double b0=5.00; double b=0.; double db=0.002;
    std::fstream fs; fs.open("Nk_11.csv", std::ios::out);
    for(int i=0; i<321; i++)
    {
        b=b0+i*db;
        fs << b << "," << Nch(b) << "," << Nkaver(b) << std::endl;
    }
    fs.close();
}
