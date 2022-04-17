#include <iostream>
using namespace std;

int main() {
    int quarters, dimes, nickels, pennies, cents, dollars;

    cout<<"Please enter the amount of money to convert:"<<endl;
    cout<<"# of dollars:"<<endl;
    cin>>dollars;
    cout<<"# of cents:"<<endl;
    cin>>cents;

    cents = (dollars*100) + cents;

    quarters = cents / 25;
    dimes = (cents % 25) / 10;
    nickels = (cents - ((quarters * 25) + (dimes * 10))) / 5;
    pennies = cents - (((quarters * 25) + (dimes * 10)) + (nickels * 5));

    cout<<"The coins are "<<quarters<<" quarters, "<<dimes<<" dimes, ";
    cout<<nickels<<" nickels and "<<pennies<<" pennies"<<endl;

    return 0;
}
