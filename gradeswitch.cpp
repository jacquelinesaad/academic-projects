/*
Author: Jacqueline Saad
Course: CSCI-136
Instructor: Tong Yi
Assignment: grade switch assignment
*/

#include <iostream>

#include<bits/stdc++.h>

using namespace std;

int main() {
  /*take the input of the score from the students...*/
  int score;
  int x;
  char letter_grade;

  cout << "Enter the score of the student\n";
  cin >> score;
  
  /*assign the code to each score in the given range*/

    if(score<60)
        x=-1;
    else if(score>=60 && score<70)
        x=-2;
    else if(score>=70 && score<80)
        x=-3;   
    else if(score>=80 && score<90)
        x=-4;
    else
        x=-5;

    /*switch assigning to a grade letter using case statement*/

    switch(x)
    {
        case -1:
            letter_grade='F';
            break;
        case -2:
            letter_grade='D';
            break;
        case -3:
            letter_grade='C';
            break;
        case -4:
            letter_grade='B';
            break;
        case -5:
            letter_grade='A';
            break;    
    //case -1=F=less than 60
    //-2=less than 70 but greater than equal to 60
    }

    //finally print the grade letter_grade
    cout<<"letter_grade is: "<<letter_grade<<endl;

return 0;
}