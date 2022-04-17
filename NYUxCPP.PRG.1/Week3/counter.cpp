#include <iostream>

int main(void)
{
    int nQuarter, nDime, nNickel, nPenny;
    int nDollar, nCent;
    double fAccount;

    // asks the user to enter the number of coins
    std::cout << "Please enter the number of coins:" << std::endl;
    std::cout << "# of quarters: "; // 25 cents
    std::cin >> nQuarter;
    std::cout << "# of dimes: "; // 10 cents
    std::cin >> nDime;
    std::cout << "# of nickels: "; // 5 cents
    std::cin >> nNickel;
    std::cout << "# of pennies: "; // 1 cent
    std::cin >> nPenny;

    // does the accounting
    fAccount = (nQuarter * 0.25) + (nDime * 0.10) + (nNickel * 0.05) + (nPenny * 0.01);
    nDollar = (int)fAccount;
    nCent = (fAccount - nDollar) * 100;

    // outputs the monetary value
    std::cout << "The total is " << nDollar << " dollars and " << nCent << " cents" << std::endl;

    return 0;
} // closes main(void)
