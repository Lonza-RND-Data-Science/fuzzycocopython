#include "tests.h"
#include "fuzzy_set.h"

using namespace fuzzy_coco;

TEST(FuzzySet, test1) {
    FuzzySet set("set1");
    EXPECT_EQ(set.getName(), "set1");
    EXPECT_TRUE(is_na(set.getPosition()));

    cerr << "FuzzySet" << set << endl;

    set.setPosition(1);
    EXPECT_EQ(set.getPosition(), 1);

    cerr << "FuzzySet" << set << endl;

    FuzzySet set2("set2", 20);
    EXPECT_EQ(set2.getName(), "set2");
    EXPECT_EQ(set2.getPosition(), 20);
    
    FuzzySet set3 = set2;
    EXPECT_EQ(set3, set2);
}

TEST(FuzzySet, load_describe) {
  FuzzySet set("set1");
  set.setPosition(MISSING_DATA_DOUBLE);
  auto desc = set.describe();
  cerr << desc;

  EXPECT_TRUE(desc.is_scalar());
  EXPECT_EQ(desc.name(), "set1");
  EXPECT_EQ(desc.scalar().get_double(), MISSING_DATA_DOUBLE);

  FuzzySet::printDescription(cerr, desc);

  FuzzySet set2 = FuzzySet::load(desc);
  EXPECT_EQ(set2, set);

  set.setPosition(-1.1);
  desc = set.describe();
  EXPECT_EQ(desc.scalar().get_double(), -1.1);
  EXPECT_EQ(FuzzySet::load(desc), set);
}