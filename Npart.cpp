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
void Npart1(); void Npart2(); void Npart3(); void Npart4(); void Npart5(); void Npart6(); void Npart7();
void Npart8(); void Npart9(); void Npart10(); void Npart11();
//===================================================================================================
int A=16; int B=16; double Ra=2.82; double Rb=2.82; int n=100;
double IXsecNN=6.80; //fm^2
//===================================================================================================
double Thpcd(double ss) // radius -> nor
{
    double B_local=0.13*16.;
    return 1./(2.*PI*B_local)*exp(-pow(ss,2)/(2*B_local));
}

//===================================================================================================
double Npart(double b)
{
    double Cx=0.; double Cy=0.; double dC=2.*Ra/n;
    double CA=0.; double CB=0.; double sumNpart=0.; double IXBn=0.;
    for(int i=0; i<n; i++)
    {
        Cx=-Ra+i*dC;
        for(int j=0; j<n; j++)
        {
            Cy=-Ra+j*dC;

            CA=sqrt(pow(Cx,2)+pow(Cy,2)); CB=sqrt(pow(Cx-b,2)+pow(Cy,2));
            IXBn=1.-pow(1.-Thpcd(CB)*IXsecNN,B);
            if(CA<Ra && CB<Rb) sumNpart+=(A*Thpcd(CA)*IXBn+B*Thpcd(CB)*Thpcd(CA)*IXsecNN)*dC*dC;
            else continue;
        }
    }
    return sumNpart;
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

    std::thread t1(Npart1); std::thread t2(Npart2); std::thread t3(Npart3); std::thread t4(Npart4); std::thread t5(Npart5);
    std::thread t6(Npart6); std::thread t7(Npart7); std::thread t8(Npart8); std::thread t9(Npart9); std::thread t10(Npart10); std::thread t11(Npart11);
 
    t1.join(); t2.join(); t3.join(); t4.join(); t5.join(); t6.join(); t7.join(); t8.join(); t9.join(); t10.join(); t11.join();
 
    std::string filename2="cmd /c type Npart_1.csv Npart_2.csv Npart_3.csv Npart_4.csv Npart_5.csv Npart_6.csv Npart_7.csv Npart_8.csv Npart_9.csv Npart_10.csv Npart_11.csv >IX_b_Npart.csv";
    system(filename2.c_str());

    std::string filename3="cmd /c del Npart_*.csv";
    system(filename3.c_str());

    auto end=sc.now();
    auto time_span=static_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Npart calculation took: " << time_span.count() << " seconds" << std::endl;

    return 0;
}
//===================================================================================================
void Npart1()
{
    double b0=0.; double b=0.; double db=0.002;
    std::fstream fs1; fs1.open("Npart_1.csv", std::ios::out);
    for(int i=0; i<250; i++)
    {
        b=b0+i*db;
        fs1 << b << "," << Npart(b)  << std::endl;
    }
    fs1.close();
}
void Npart2()
{
    double b0=0.50; double b=0.; double db=0.002;
    std::fstream fs1; fs1.open("Npart_2.csv", std::ios::out);
    for(int i=0; i<250; i++)
    {
        b=b0+i*db;
        fs1 << b << "," << Npart(b)  << std::endl;
    }
    fs1.close();
}
void Npart3()
{
    double b0=1.00; double b=0.; double db=0.002;
    std::fstream fs1; fs1.open("Npart_3.csv", std::ios::out);
    for(int i=0; i<250; i++)
    {
        b=b0+i*db;
        fs1 << b << "," << Npart(b)  << std::endl;
    }
    fs1.close();
}
void Npart4()
{
    double b0=1.50; double b=0.; double db=0.002;
    std::fstream fs1; fs1.open("Npart_4.csv", std::ios::out);
    for(int i=0; i<250; i++)
    {
        b=b0+i*db;
        fs1 << b << "," << Npart(b)  << std::endl;
    }
    fs1.close();
}
void Npart5()
{
    double b0=2.00; double b=0.; double db=0.002;
    std::fstream fs1; fs1.open("Npart_5.csv", std::ios::out);
    for(int i=0; i<250; i++)
    {
        b=b0+i*db;
        fs1 << b << "," << Npart(b)  << std::endl;
    }
    fs1.close();
}
void Npart6()
{
    double b0=2.50; double b=0.; double db=0.002;
    std::fstream fs1; fs1.open("Npart_6.csv", std::ios::out);
    for(int i=0; i<250; i++)
    {
        b=b0+i*db;
        fs1 << b << "," << Npart(b)  << std::endl;
    }
    fs1.close();
}
void Npart7()
{
    double b0=3.00; double b=0.; double db=0.002;
    std::fstream fs1; fs1.open("Npart_7.csv", std::ios::out);
    for(int i=0; i<250; i++)
    {
        b=b0+i*db;
        fs1 << b << "," << Npart(b)  << std::endl;
    }
    fs1.close();
}
void Npart8()
{
    double b0=3.50; double b=0.; double db=0.002;
    std::fstream fs1; fs1.open("Npart_8.csv", std::ios::out);
    for(int i=0; i<250; i++)
    {
        b=b0+i*db;
        fs1 << b << "," << Npart(b)  << std::endl;
    }
    fs1.close();
}
void Npart9()
{
    double b0=4.00; double b=0.; double db=0.002;
    std::fstream fs1; fs1.open("Npart_9.csv", std::ios::out);
    for(int i=0; i<250; i++)
    {
        b=b0+i*db;
        fs1 << b << "," << Npart(b)  << std::endl;
    }
    fs1.close();
}
void Npart10()
{
    double b0=4.50; double b=0.; double db=0.002;
    std::fstream fs1; fs1.open("Npart_10.csv", std::ios::out);
    for(int i=0; i<250; i++)
    {
        b=b0+i*db;
        fs1 << b << "," << Npart(b)  << std::endl;
    }
    fs1.close();
}
void Npart11()
{
    double b0=5.00; double b=0.; double db=0.002;
    std::fstream fs1; fs1.open("Npart_11.csv", std::ios::out);
    for(int i=0; i<250; i++)
    {
        b=b0+i*db;
        fs1 << b << "," << Npart(b)  << std::endl;
    }
    fs1.close();
}
