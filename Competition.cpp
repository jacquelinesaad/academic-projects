#include "Competition.h"
#include <iostream>
#include <cstdlib>
#include <iomanip>
using namespace std;

Competition::Competition() {
  srand(1);
  rabbit = new Hare;
  tor = new Tortoise;
  rd = new Road;
}

Competition::Competition(Hare *rabbit, Tortoise *tor, int length) {
  srand(1);
  this -> rabbit = new Hare(rabbit -> getPattern(),rabbit -> getPatternLength(),
                  rabbit -> getPosition());
  this -> tor = new Tortoise(tor -> getPattern(), tor -> getPatternLength(),
                tor -> getPosition());

  rd = new Road(length);
}

Competition::~Competition() {
  delete rabbit;
  rabbit = 0;
  delete tor;
  tor = 0;
  delete rd;
  rd = 0;
}

void Competition::start()  {
  int lastBlock = rd -> getLastBlock();
  int round = 1;
  while (rabbit -> getPosition() < lastBlock && tor -> getPosition() < lastBlock) {
    rabbit -> move();
    if (rabbit -> getPosition() < 0 ) {
      rabbit -> setPosition(0);
    }
    else if (rabbit -> getPosition() > lastBlock) {
      rabbit -> setPosition(lastBlock);
    }

    tor -> move();
    if (tor -> getPosition() < 0 ) {
      tor -> setPosition(0);
    }
    else if (tor -> getPosition() > lastBlock) {
      tor -> setPosition(lastBlock);
    }

    rd -> mark(rabbit -> getPosition(), 'H');
    rd -> mark(tor -> getPosition(), 'T');

    cout << setw(3) << round << " ";
    cout << rd -> toString() << endl;

    if (rabbit -> getPosition() == tor -> getPosition()) {
      cout << "Ouch, Tortoise bites Hare." << endl;
    }

    rd -> mark(tor -> getPosition(), ' ');
    rd -> mark(rabbit -> getPosition(), ' ');
    round++;
  }

  if (rabbit -> getPosition() == lastBlock && tor -> getPosition() == lastBlock) {
    cout << "It is a tie." << endl;
  }
  else if (rabbit -> getPosition() == lastBlock) {
    cout << "Yuck, hare wins." << endl;
  }
  else if (tor -> getPosition() == lastBlock) {
      cout << "Yay!!! tortoise wins!" << endl;
  }
}
