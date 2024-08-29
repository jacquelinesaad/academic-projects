/*
Author: Jacqueline Saad
Course: CSCI-136
Instructor: Tong Yi
Assignment: HW E9.3


Simulate a circuit for controlling a hallway light that has switches at both ends of the hallway.
Each switch can be up or down, and the light can be on or off.
Toggling either switch turns the lamp on or off. Provide member functions
*/

#include<iostream>

using namespace std;

class Circuit{
 //instance variables to store switch states and lamp state
    int switch1;
    int switch2;
    int lamp;
public:
//constructor
Circuit(){
//by default, switches are down (off) and lamp is off
    switch1=0;
    switch2=0;
    lamp=0;
}

//returns the state of first switch
int get_first_switch_state(){
    return switch1;
}

//returns the state of second switch
int get_second_switch_state(){
    return switch2;
}

//returns the state of lamp
int get_lamp_state(){
    return lamp;
}

//toggles first switch
void toggle_first_switch(){
    //if switch1's value is 1, changing to 0, if it is 0, changing to 1
    switch1=(switch1==1?0:1);
    //doing the same for lamp state
    lamp=(lamp==1?0:1);
}

//toggles second switch
void toggle_second_switch(){
    //if switch2's value is 1, changing to 0, if it is 0, changing to 1
    switch2=(switch2==1?0:1);
    //doing the same for lamp state
    lamp=(lamp==1?0:1);
 }
};

int main(){
    //creating a Circuit
    Circuit c;

    //displaying initial state
    cout<<"Initial state:"<<endl;
    cout<<"Switch1: "<<c.get_first_switch_state()<<endl;
    cout<<"Switch2: "<<c.get_second_switch_state()<<endl;
    cout<<"Lamp: "<<c.get_lamp_state()<<endl;

    //toggling first switch, lamp should be on by now
    cout<<"Toggling switch1..."<<endl;
    c.toggle_first_switch();

    //displaying states.
    cout<<"Switch1: "<<c.get_first_switch_state()<<endl;
    cout<<"Switch2: "<<c.get_second_switch_state()<<endl;
    cout<<"Lamp: "<<c.get_lamp_state()<<endl;

    //toggling second switch, lamp should be off by now
    cout<<"Toggling switch2..."<<endl;
    c.toggle_second_switch();

    //displaying states
    cout<<"Switch1: "<<c.get_first_switch_state()<<endl;
    cout<<"Switch2: "<<c.get_second_switch_state()<<endl;
    cout<<"Lamp: "<<c.get_lamp_state()<<endl;

    return 0;
}