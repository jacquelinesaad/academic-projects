/*
Author: Jacqueline Saad
Course: CSCI-136
Instructor: Tong Yi
Assignment: Project 2B

The user will give an int in the inputted ranges.
The computer will keep guessing the middle number of 
the range based on feedback until correct int is given. 
*/

#include <iostream>
using namespace std;

int main()
{
	int tries = 1, guess, answer, leftrange, rightrange, feedback; //set up needed variables

	//user inputs ranges
	cout << "Enter left end in range: " << endl;
	cin >> leftrange;
	cout << "Enter right end in range: " << endl;
	cin >> rightrange;

	//check and make sure for valid ranges
	while (leftrange > rightrange)
    
		{
			cout << "Range is invalid, please re-enter: " << endl;
			cout << "Enter left end in range: " << endl;
			cin >> leftrange;
			cout << "Enter right end in range: " << endl;
			cin >> rightrange;
		}

    //user chooses an int in that range
    cout << "Enter the answer: ";
    cin >> answer;
    
    //checks for answer within range
    while (answer < leftrange || answer > rightrange)
		{
			cout << "Answer not in range, please re-enter: ";
			cin >> answer;
		}

	cout << "User has an int in [" << leftrange << ", " << rightrange << "]. Computer will guess." << endl;

    //computer makes a guess that equals the middle of the overall range

    guess = (rightrange + leftrange)/2;
    cout << "Guess #1: " << guess << endl;

    //user gives a feedback for each guess
    while (leftrange != rightrange && guess != answer)
    {
        cout << "How is my guess?" << endl << "1. too big   2. too small    3. just right";
        cout << endl << "Enter only 1, 2, or 3: ";
        cin >> feedback; 
        tries ++; //increment tries

        //checks feedback value and makes sure feedback is valid
        while (feedback < 1 || feedback > 3)
        {
            cout << "How is my guess?" << endl << "1. too big   2. too small    3. just right";
            cout << endl << "Enter only 1, 2, or 3: ";
            cin >> feedback;
        }

        if (feedback == 1)
        {
            cout << "Guess was too big!" << endl;
            rightrange = guess;
        }

        else if (feedback == 2)
        {
            cout << "Guess was too small!" <<endl;
            leftrange = guess;
        }

        else if (feedback == 3)
        {
            //the game ends when the computer makes a correct guess

            if (guess != answer)
            {
                cout << "Not exactly, that guess is not correct." << endl;
            }
            else

            {
                cout << "Guess #" << tries << " The answer must be " << guess << "." << endl;
            }
        }

    //computer guess is altered based on feedback
        guess = (rightrange + leftrange) / 2;
        cout << "Guess #" << tries << ": " << guess << endl;
    }

    //checks if answer is correct or any input given was wrong
    if (guess == answer)
    {
        cout << "Guess #" << tries << ": The answer must be " << guess << "." << endl;
    }

    else
    {
        cout << "Something went wrong. Please try again!" << endl;
    }

    return 0;
}