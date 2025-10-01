#include "tests.h"
#include "string_utils.h"

using namespace fuzzy_coco;
using namespace StringUtils;


string CONTENT_REF = 
 R"({
    "Temperature": {
       "Cold": 10,
       "Warm": 20.1,
       "Hot":  30
    }
  }
)";

string CONTENT_COMMENTED = R"(
{
# comment1 starting at col 1
    "Temperature": {
      # comment2
       "Cold": 10, # inline comment 3
       "Warm": 20.1,
       # "toto": "titi" commented line
       "Hot":  30
    }
       # comment4
  }# inline comment 5 )";

TEST(StringUtils, stripComments) {
  EXPECT_EQ(stripComments(CONTENT_REF), CONTENT_REF);
  auto stripped = stripComments(CONTENT_COMMENTED);
  // cerr << stripped;
  EXPECT_EQ(stripped, CONTENT_REF);
  EXPECT_EQ(stripComments(stripped), CONTENT_REF);

}

TEST(StringUtils, prettyDistinguishableDoubleToString) {
  EXPECT_EQ(prettyDistinguishableDoubleToString(1), "1.0");
  EXPECT_EQ(prettyDistinguishableDoubleToString(1.01), "1.01");
  EXPECT_EQ(prettyDistinguishableDoubleToString(1.0100), "1.01");
  EXPECT_EQ(prettyDistinguishableDoubleToString(10), "10.0");
  EXPECT_EQ(prettyDistinguishableDoubleToString(0), "0.0");

  EXPECT_EQ(prettyDistinguishableDoubleToString(0.10000001), "0.1");

}
