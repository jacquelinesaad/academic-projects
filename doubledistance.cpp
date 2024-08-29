/*
Author: Jacqueline Saad
Course: CSCI-136
Instructor: Tong Yi
Assignment: hw 7.16

Define a structure Point. 
A point has an x- and a y-coordinate. Write a function double distance(Point a, Point b)
that computes the distance from a to b. Write a program that reads the coordinates of the
points, calls your function, and displays the result.
*/


#include<iostream>
#include<cstdlib>
#include<string>
#include<vector>
#include<cmath>
using namespace std;

//double distance(Point a, Point b);
//void print_coord(Point coord);

struct Point
{
public:
    double x;
    double y;
};

double distance(Point a, Point b)
{
    double dist;

    dist = sqrt((pow((b.x - a.x), 2) + pow((b.y - a.y), 2)));
    return dist;
}

void print_coord(Point coord)
{
    cout << "(" << coord.x << ", " << coord.y << ")" << endl;
}

int main()
{
    double x1, x2, y1, y2, total;

    cout << "Enter first coordinates. " << endl;
    cout << "x: ";
    cin >> x1; 
    cout << "y: ";
    cin >> y1;
    cout << endl;

    cout << "Enter second coordinates. " << endl;
    cout << "x: ";
    cin >> x2; 
    cout << "y: ";
    cin >> y2;
    cout << endl;

    Point a;
    a.x = x1;
    a.y = y1;

    Point b;
    b.x = x2;
    b.y = y2;

    cout << "a: ";
    print_coord(a);

    cout << "b: ";
    print_coord(b);

    cout << endl;

    total = distance(a, b);

    cout << "The distance between point a and point b is: " << total << endl;
    cout << endl;
}