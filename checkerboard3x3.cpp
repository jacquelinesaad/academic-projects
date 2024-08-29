/*
Author: Jacqueline Saad
Course: CSCI-136
Instructor: Tong Yi
Assignment: Lab 4 G

checkerboard3x3.cpp
*/

#include <iostream>
using namespace std;

int main()
{ 
    int width, height;

    cout << "Enter width " << endl; //have user enter width of box
    cin >> width;
    cout << "Enter length" << endl; //have user enter length of box
    cin >> height;

    //go with the row first, then with column

    for(int row  = 0; row < height; row++) {
        for (int col = 0; col < width; col++){
            //print out the stars whenever both row and col are even or odd
            if ((row/3+col/3)%2 == 0) 
            {
            cout << "*";
            }
            else //print spaces
            {
            cout << " ";
            }
        }
     cout  << endl;
    }
return 0;
}