#include "tests.h"
#include "types.h"

using namespace fuzzy_coco;

TEST(types, is_na) {
  EXPECT_TRUE(is_na(MISSING_DATA_DOUBLE));
  EXPECT_TRUE(is_na(NA_D));
  EXPECT_TRUE(is_na(MISSING_DATA_INT));
  EXPECT_TRUE(is_na(NA_I));

  EXPECT_FALSE(is_na(1.1));
  EXPECT_FALSE(is_na(0));
}

