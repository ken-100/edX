#include <iostream>
#include <cmath>
#include <iomanip>

int main(void)
{
    double dWeight, dHeight, dBMI;

    // asks the user to enter the number of coins
    std::cout << "Please enter weight in kilograms: ";
    std::cin >> dWeight;
    std::cout << "Please enter height in meters: ";
    std::cin >> dHeight;

    // BMI calculation
    dBMI = dWeight / pow(dHeight, 2);

    // outputs BMI
    std::cout << std::fixed;
    std::cout << std::setprecision(2);
    std::cout << "BMI is" << ": " << dBMI << std::endl;

    return 0;
} // closes main(void)
