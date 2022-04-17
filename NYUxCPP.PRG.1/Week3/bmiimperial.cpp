#include <iostream>
#include <cmath>
#include <iomanip>

const double POUND_KG = 0.453592;
const double INCH_M = 0.0254;
int main(void)
{
    double dWeight, dHeight, dBMI;

    // asks the user to enter the number of coins
    std::cout << "Please enter weight in pounds: ";
    std::cin >> dWeight;
    std::cout << "Please enter height in inches: ";
    std::cin >> dHeight;

    // BMI calculation
    dBMI = (dWeight * POUND_KG) / pow((dHeight * INCH_M), 2);

    // outputs BMI
    std::cout << std::fixed;
    std::cout << std::setprecision(2);
    std::cout << "BMI is: " << dBMI << std::endl;

    return 0;
} // closes main(void)
