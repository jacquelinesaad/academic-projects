/*
Author: Jacqueline Saad
Course: CSCI-136
Instructor: Tong Yi
Assignment: Project 2C

The user will give an int in the inputted ranges.
The computer will keep guessing a random nunmber of 
the range based on feedback until correct int is given. 
*/

#include<iostream>
#include<cstdlib>
#include<ctime>

using namespace std;

int main()
{
	int leftrange, rightrange, random, answer, feedback, tries; //set up needed variables

    //user inputs ranges
	cout << endl;
	cout << "Enter the left end of the range: ";
	cin >> leftrange;
	cout << "Enter the right end of the range: ";
	cin >> rightrange;
	cout << endl;

    //check for valid range
	while (leftrange > rightrange)
	{
		cout << "Range is invalid. Enter new range." << endl;
		cout << endl;
		cout << "Enter the left end of the range: " << endl;
		cin >> leftrange;
		cout << "Enter the right end of the range: " << endl;
		cin >> rightrange;
		cout << endl;
	}

    //user enters the answer
	cout << "Enter the answer: ";
	cin >> answer;
	cout << endl;

    //makes sure answer is in range
	while (answer < leftrange || answer > rightrange)
	{
		cout << "Number is invalid. Enter new number." << endl;
		cout << "Enter the answer: ";
		cin >> answer;
		cout << endl;
	}

    //starts the computer game
	cout << "User has an int in [" << leftrange << ", " << rightrange << "]. Computer will guess." << endl;

    //changes from 2b, now a random guess rather than median
	unsigned seed = time(0);
	srand(seed);
	random = (rand() % (rightrange - leftrange + 1)) + leftrange;
	tries = 1; //start tries at one

	cout << endl;

    //while the random guess is not equal to the answer
	while (random != answer)
	{
		cout << "Guess #" << tries << ": " << random << ".";
		cout << " How is my guess?" << endl;
		cout << "1. too big   2. too small    3. just right" << endl;
		cout << "Enter only 1, 2, or 3: ";
		cin >> feedback;
		cout << endl;

		while (feedback < 1 || feedback > 3) //makes sure feedback is valid
		{
			cout << "Invalid Feedback. " << endl;
			cout << "Enter correct feedback, (1, 2, or 3): ";
			cin >> feedback;
			cout << endl;
		}

        //user inputs too big !!
        //but incorrectly
		if (feedback == 1)
		{				
			if (random < answer)
			{	
				cout << "Incorrect Feedback. (Feedback: 2) " << endl;
				cout << "Guess was too small." << endl;
				feedback = 2;
				tries++;
				leftrange = random + 1;
				random = (rand() % (rightrange - leftrange + 1)) + leftrange;			
			}
        //correctly
			else if (random > answer)
			{
				tries++;
				cout << "Guess was too big." << endl;
				cout << endl;
				rightrange = random - 1;
				random = (rand() % (rightrange - leftrange + 1)) + leftrange;		
			}
		}

        //user inputs too small !!
        //but incorrectly
		else if (feedback == 2)
		{
			if (random > answer)
			{
				cout << "Incorrect Feedback. (Feedback: 1)" << endl;
				cout << "Guess was too big." << endl;
				feedback = 1;	
				tries++;
				rightrange = random - 1;
				random = (rand() % (rightrange - leftrange + 1)) + leftrange;
			}
        //correctly
			else if (random < answer)
			{
				tries++;
				cout << "Guess was too small." << endl;
				cout << endl;
				leftrange = random + 1;
				random = (rand() % (rightrange - leftrange + 1)) + leftrange;	
			}				
		}

        //user inputs just right !!
        //incorrectly
		else if (feedback == 3)
		{
			if (random < answer)
			{
				cout << "Incorrect Feedback. (Feedback: 2)" << endl;
				cout << endl;
				cout << "Guess is not correct." << endl;
				cout << endl;
				feedback = 2;
				tries++;
				leftrange = random + 1;
				random = (rand() % (rightrange - leftrange + 1)) + leftrange;
			}

			else if(random > answer)
			{
				cout << "Incorrect Feedback. (Feedback: 1)" << endl;
				cout << endl;
				cout << "Guess was too big." << endl;
				cout << endl;
				feedback = 1;	
				tries++;
				leftrange = random - 1;
				random = (rand() % (rightrange - leftrange + 1)) + leftrange;
			}
            //correctly
			else
			{
				cout << "Guess #" << tries << ": ";
				cout << "The answer must be " << random << "." << endl;
			}	
		}
	}

	int gap = rightrange - leftrange;

	if (gap <= 2)
	{
		cout << "Guess #" << tries << ": ";
		cout << "The answer must be " << random << "." << endl;
	}
	else
	{
		cout << "Guess #" << tries << ": " << random << ".";
		cout << " How is my guess?" << endl;
		cout << "1. too big   2. too small    3. just right" << endl;
		cout << "Enter only 1, 2, or 3: ";
		cin >> feedback;
		cout << endl;

		if (feedback == 3)
		{
			cout << "Guess #" << tries << ": " <<  "The answer is " << random << "." << endl;
		}
		else
		{
			cout << "Incorrect Feedback. (Feedback: 3)" << endl;
			cout << endl;
			cout << "Guess #" << tries << ": " <<  "The answer is " << random << "." << endl;
		}
	}

	cout << endl;

	return 0;
}