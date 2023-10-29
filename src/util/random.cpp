
#include <iostream>

using namespace std;

double get_random()
{
    // return random number between -1 and 1
    return (rand() / (double)RAND_MAX) * 2 - 1;
}