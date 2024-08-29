/*
Author: Jacqueline Saad
Course: CSCI-136
Instructor: Tong Yi
Assignment: Project 2A
*/

#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

int main()
{
    //Enter the left end and right end of a range of integers.
    int lower, higher, guess;
    cout << "Enter the lower bound: ";
    cin >> lower;
    cout << "Enter the higher bound: ";
    cin >> higher;

    //Making sure bounds are usable. 
    while (lower>=higher)
    {
        cout << "Bounds invalid, re-enter: " << endl;
        cout << "Enter the lower bound: ";
        cin >> lower;
        cout << "Enter the lower bound: ";
        cin >> higher;
    }

    //Computer generates a random integer in the range.
    srand((unsigned)time(0));
    int random = rand() % higher + lower;
    cout << endl << "Enter your guess: ";
    cin >> guess;

    //User makes a guess and computer will give feedback.
    //Based on the feedback user will continue to enter 
    //guesses until the guess equals the answer.
    while(guess != random)
    {
        //Checks if guess is out of bounds. 
        while (guess < lower || guess > higher)
        {
            cout << "The guess is out of range. Re-enter: " << endl;
            cin >> guess;
        }
        if (guess < random)
        {
            cout << "guess is too small" << endl;
            cout << "Re-enter your guess: ";
            cin >> guess;
        }
        if (guess > random)
        {
            cout << "guess is too big" << endl;
            cout << "Re-enter your guess: ";
            cin >> guess;
        }
    }
    //Output when guess is correct. 
    cout << "Congratulations!" << endl;
    return 0;
}