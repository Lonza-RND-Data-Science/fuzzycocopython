
#ifndef FUZZY_OPERATOR_H
#define FUZZY_OPERATOR_H

namespace fuzzy_coco {

class FuzzyOperator
{
public:
    FuzzyOperator() {}
    virtual ~FuzzyOperator() {}

    virtual double operate(double x, double y) =0;
};

class FuzzyOperatorAND : public FuzzyOperator
{
public:
    FuzzyOperatorAND() {}
    virtual ~FuzzyOperatorAND() {}
    // AND operator --> use min(x, y) unless x or y is the dontcare
    virtual double operate(double x, double y) {
      double res = min(x, y);
       // DONTCARE VALUE < 0; If the min < 0 ==> x or/and y are < 0 --> return the other (the max) then
       // if both qre < 0 --> < 0
      return res >= 0 ? res : max(x, y); 
    }
};

}
#endif // FUZZY_OPERATOR_H
