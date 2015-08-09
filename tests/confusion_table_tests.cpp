#include "test.h"
#include "vi/nn/confusion_table.h"

TEST(confusion_table, calculates_performance_measures) {
  using vi::nn::confusion_table;
  confusion_table a(2, 0, 0, 0);
  EXPECT_FLOAT_EQ(1.0, a.accuracy());
  EXPECT_FLOAT_EQ(0.0, a.error_rate());
  EXPECT_FLOAT_EQ(1.0, a.precision());
  EXPECT_FLOAT_EQ(1.0, a.recall());
  EXPECT_FLOAT_EQ(1.0, a.fscore());
  EXPECT_FLOAT_EQ(0.0, a.specificity());
  EXPECT_FLOAT_EQ(0.5, a.auc());

  confusion_table b(2, 2, 0, 0);
  EXPECT_FLOAT_EQ(0.5, b.accuracy());
  EXPECT_FLOAT_EQ(0.5, b.error_rate());
  EXPECT_FLOAT_EQ(1.0, b.precision());
  EXPECT_FLOAT_EQ(0.5, b.recall());
  EXPECT_FLOAT_EQ(2.0 / 3.0, b.fscore());
  EXPECT_FLOAT_EQ(0.0, b.specificity());
  EXPECT_FLOAT_EQ(0.25, b.auc());

  confusion_table c(2, 2, 2, 0);
  EXPECT_FLOAT_EQ(1.0 / 3.0, c.accuracy());
  EXPECT_FLOAT_EQ(2.0 / 3.0, c.error_rate());
  EXPECT_FLOAT_EQ(0.5, c.precision());
  EXPECT_FLOAT_EQ(0.5, c.recall());
  EXPECT_FLOAT_EQ(0.5, c.fscore());
  EXPECT_FLOAT_EQ(0.0, c.specificity());
  EXPECT_FLOAT_EQ(0.25, c.auc());

  confusion_table d(2, 2, 2, 2);
  EXPECT_FLOAT_EQ(0.5, d.accuracy());
  EXPECT_FLOAT_EQ(0.5, d.error_rate());
  EXPECT_FLOAT_EQ(0.5, d.precision());
  EXPECT_FLOAT_EQ(0.5, d.recall());
  EXPECT_FLOAT_EQ(0.5, d.fscore());
  EXPECT_FLOAT_EQ(0.5, d.specificity());
  EXPECT_FLOAT_EQ(0.5, d.auc());

  confusion_table e(0, 0, 0, 0);
  EXPECT_FLOAT_EQ(0.0, e.accuracy());
  EXPECT_FLOAT_EQ(0.0, e.error_rate());
  EXPECT_FLOAT_EQ(0.0, e.precision());
  EXPECT_FLOAT_EQ(0.0, e.recall());
  EXPECT_FLOAT_EQ(0.0, e.fscore());
  EXPECT_FLOAT_EQ(0.0, e.specificity());
  EXPECT_FLOAT_EQ(0.0, e.auc());
}
