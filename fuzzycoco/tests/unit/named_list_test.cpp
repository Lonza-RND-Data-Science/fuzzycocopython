#include "tests.h"
#include "types.h"
#include "named_list.h"
#include "string_utils.h"
#include "logging_logger.h"

using namespace fuzzy_coco;
using namespace StringUtils;
using namespace logging;

TEST(Scalar, parse) {
  {
    string content = "1";
    auto scalar = Scalar::parse(content);
    EXPECT_TRUE(scalar.is_int());
    EXPECT_EQ(scalar.get_int(), 1);
  }

  {
    string content = "-1.001";
    auto scalar = Scalar::parse(content);
    EXPECT_TRUE(scalar.is_double());
    EXPECT_EQ(scalar.get_double(), -1.001);
  }

  {
    string content = R"("toto")";
    auto scalar = Scalar::parse(content);
    EXPECT_TRUE(scalar.is_string());
    EXPECT_EQ(scalar.get_string(), "toto");
  }

  // ========== edge cases ========
  {
    string content = "1}";
    auto scalar = Scalar::parse(content);
    EXPECT_EQ(scalar.get_int(), 1);
  }

  {
    string content = R"("toto"})";
    auto scalar = Scalar::parse(content);
    EXPECT_EQ(scalar.get_string(), "toto");
  }
}

TEST(NamedList, basic) {
  Scalar s1(2.0);
  cerr << s1 << endl;

  NamedList list;
  EXPECT_FALSE(list.is_scalar());
  EXPECT_TRUE(list.is_list());
  EXPECT_TRUE(list.empty());
  EXPECT_TRUE(list.name().empty());
  EXPECT_TRUE(list.scalar().is_null());

  EXPECT_FALSE(list.has("nb"));
  list.add("nb", 2);
  EXPECT_TRUE(list.has("nb"));
  EXPECT_FALSE(list.is_scalar());
  EXPECT_TRUE(list.is_list());
  EXPECT_EQ(list.size(), 1);
  EXPECT_FALSE(list.empty());
  EXPECT_TRUE(list.name().empty());
  EXPECT_TRUE(list.scalar().is_null());
  const auto& elt = list[0];
  EXPECT_EQ(elt.name(), "nb");
  EXPECT_EQ(elt.scalar(), 2);

  list.add("nb2", 3);
  EXPECT_EQ(list.size(), 2);
  {
    const auto& elt = list[1];
    EXPECT_EQ(elt.name(), "nb2");
    EXPECT_EQ(elt.scalar(), 3);
  }

  // double
  list.add("pi", 3.1416);
  EXPECT_TRUE(list.has("pi"));
  EXPECT_EQ(list.size(), 3);
  {
    const auto& elt = list[2];
    EXPECT_EQ(elt.name(), "pi");
    EXPECT_EQ(elt.scalar(), 3.1416);
  }

  // string
  list.add("experiment", "fuzzy coco");
  EXPECT_EQ(list.size(), 4);
  {
    const auto& elt = list[3];
    EXPECT_EQ(elt.name(), "experiment");
    EXPECT_EQ(elt.scalar().get_string(), "fuzzy coco");
  }

  // bool
  list.add("bool", true);
  EXPECT_EQ(list.size(), 5);
  {
    const auto& elt = list[4];
    EXPECT_TRUE(elt.scalar().is_bool());
    EXPECT_EQ(elt.name(), "bool");

    EXPECT_EQ(elt.scalar().get_bool(), true);
  }

  cerr << list;
  list.add("sublist", list);
  EXPECT_EQ(list.size(), 6);
  {
    const auto& elt = list[5];
    EXPECT_EQ(elt.name(), "sublist");
    EXPECT_FALSE(elt.is_scalar());
    EXPECT_EQ(elt.size(), 5);
  }

  // ==== get ====
  EXPECT_EQ(list.fetch("nb").scalar(), 2);
  EXPECT_EQ(list.fetch("nb2").scalar(), 3);
  EXPECT_EQ(list.fetch("pi").scalar(), 3.1416);
  EXPECT_EQ(list.fetch("bool").scalar().get_bool(), true);
  EXPECT_EQ(list.fetch("experiment").scalar(), "fuzzy coco");
  EXPECT_EQ(list.fetch("sublist").size(), 5);
  EXPECT_THROW(list.fetch("do not exist"), runtime_error);

  // dups
  list.add("dup", 1);
  list.add("dup", 2);
  EXPECT_EQ(list.fetch("dup").scalar(), 1);
  EXPECT_EQ(list[7].scalar(), 2);
  cerr << list;

  // ==== names ===
  EXPECT_EQ(list.names(), vector<string>({"nb", "nb2", "pi", "experiment", "bool", "sublist", "dup", "dup"}));
  
  //  range-based for loo
  for (auto elt : list) {
    EXPECT_TRUE(elt.is_list() || elt.is_scalar());
  }

  // get_list
  EXPECT_EQ(list.get_list("sublist").name(), "sublist");
  EXPECT_EQ(list.get_list("NO_KEY", NamedList()), NamedList());

  // get_string
  EXPECT_EQ(list.get_string("experiment"), "fuzzy coco");
  EXPECT_EQ(list.get_string("experiment", "default"), "fuzzy coco");
  EXPECT_EQ(list.get_string("toto", "default"), "default");

  EXPECT_THROW(list.get_string("nb"), bad_variant_access);
  EXPECT_EQ(list.get_int("nb"), 2);
  EXPECT_EQ(list.get_int("nb", -1), 2);
  EXPECT_EQ(list.get_int("toto", -1), -1);
  EXPECT_EQ(list.get_bool("bool"), true);
  EXPECT_EQ(list.get_bool("bool", false), true);
  EXPECT_EQ(list.get_bool("toto", false), false);
  EXPECT_EQ(list.get_double("pi"), 3.1416);
  EXPECT_EQ(list.get_double("pi", 0.1), 3.1416);
  EXPECT_EQ(list.get_double("toto", 0.1), 0.1);
}

TEST(NamedList, get_as_int) {
  NamedList list;

  list.add("good", 2.0);
  list.add("bad", 2.1);
  list.add("int", 2);

  EXPECT_EQ(list.get_as_int("int"), 2);
  EXPECT_EQ(list.get_as_int("good"), 2);
  EXPECT_THROW(list.get_as_int("bad"), runtime_error);

  EXPECT_EQ(list.get_as_int("int", 0), 2);
  EXPECT_EQ(list.get_as_int("noway", 0), 0);
  EXPECT_EQ(list.get_as_int("good", 0), 2);
  EXPECT_THROW(list.get_as_int("bad", 0), runtime_error);
}

TEST(Scalar, basic) {
  { // null
    Scalar s;
    EXPECT_TRUE(s.is_null());
    EXPECT_FALSE(s.is_string() || s.is_double() || s.is_int());
    EXPECT_THROW(s.get_string(), bad_variant_access);
    EXPECT_FALSE(s == "titi" || s == 1 || s == 1.2);

  }
  {  // string
    Scalar s("toto");
    EXPECT_TRUE(s.is_string());
    EXPECT_FALSE(s.is_null() || s.is_double() || s.is_int());
    EXPECT_THROW(s.get_int(), bad_variant_access);

    EXPECT_EQ(s.get_string(), "toto");
    EXPECT_TRUE(s == "toto");
    EXPECT_FALSE(s == "titi");
    EXPECT_FALSE(s == 1 || s == 1.2);
  }
  { // int
    Scalar s(0);
    EXPECT_TRUE(s.is_int());
    EXPECT_FALSE(s.is_null() || s.is_double() || s.is_string());
    EXPECT_THROW(s.get_double(), bad_variant_access);

    EXPECT_EQ(s.get_int(), 0);
    EXPECT_TRUE(s == 0);
    EXPECT_FALSE(s == 1);
    EXPECT_FALSE(s == "toto" || s == 1.2);
  }
  { // double
    Scalar s(3.14);
    EXPECT_TRUE(s.is_double());
    EXPECT_FALSE(s.is_null() || s.is_int() || s.is_string());
    EXPECT_THROW(s.get_int(), bad_variant_access);

    EXPECT_EQ(s.get_double(), 3.14);
    EXPECT_TRUE(s == 3.14);
    EXPECT_FALSE(s == 1.0);
    EXPECT_FALSE(s == "toto" || s == 1);
  }
  { // bool
    Scalar s(true);
    EXPECT_TRUE(s.is_bool());
    EXPECT_FALSE(s.is_null() || s.is_int() || s.is_string() || s.is_double());
    EXPECT_THROW(s.get_int(), bad_variant_access);

    EXPECT_EQ(s.get_bool(), true);
    // EXPECT_TRUE(s == true);
    EXPECT_FALSE(s == false);
    EXPECT_FALSE(s == "toto" || s == 1 || s == 3.14);
  }
}

TEST(Scalar, get_as_int) {
  NamedList list;

  Scalar good(2.0);
  Scalar bad(2.1);
  Scalar already_int(1);

  EXPECT_EQ(already_int.get_as_int(), 1);
  EXPECT_EQ(good.get_as_int(), 2);
  EXPECT_THROW(bad.get_as_int(), runtime_error);
}

TEST(Scalar, IO) {
    // string
  {
    stringstream io;
    Scalar s("string");
    io << s;
    auto s2 = Scalar::parse(io);
    EXPECT_EQ(s2, s);
  }

  // badstring: without quote, not starting with t or f
  { 
    stringstream io;
    io << "bad";
    EXPECT_THROW(Scalar::parse(io), invalid_argument); // thrown by stoi()
  }

  // int
  {
    stringstream io;
    Scalar s(1);
    io << s;
    auto s2 = Scalar::parse(io);
    EXPECT_EQ(s2, s);
  }

  // double
  {
    stringstream io;
    Scalar s(1.1010);
    io << s;
    auto s2 = Scalar::parse(io);
    EXPECT_EQ(s2, s);
  }

  { // double: but holding an integer value
    stringstream io;
    Scalar s(1.0);
    io << s;
    auto s2 = Scalar::parse(io);
    EXPECT_EQ(s2, s);
  }

  { // bool
    stringstream io;
    Scalar s(false);
    io << s;

    auto s2 = Scalar::parse(io);
    // cerr << s2;

    EXPECT_EQ(s2.get_bool(), false);
  }

  // bad bool <=> string without quote starting with t or f
  { // bool
    stringstream io;
    io << "foufou";
    EXPECT_THROW(Scalar::parse(io), runtime_error);
  }

  // null
  {
    stringstream io;
    Scalar s;
    io << s;
    EXPECT_EQ(io.str(), "");
    auto s2 = Scalar::parse(io);
    EXPECT_EQ(s2, s);
  }

}

TEST(Scalar, IO_and_NA) {
  { // MISSING_DATA_DOUBLE
    stringstream io;
    Scalar s(MISSING_DATA_DOUBLE);
    io << s;

    auto s2 = Scalar::parse(io);

    EXPECT_EQ(s2, s);
  }

  { // MISSING_DATA_INT
    stringstream io;
    Scalar s(MISSING_DATA_INT);
    io << s;
    auto s2 = Scalar::parse(io);
    EXPECT_EQ(s2, s);

    cerr  << s << endl;
    cerr  << s2 << endl;
  }
// abort();
}

TEST(NamedList, IO) {
  // empty stream
  {
    stringstream io;
    auto l = NamedList::parse(io);
    EXPECT_EQ(l, NamedList());
  }


    // empty list
  {
    stringstream io;
    NamedList l;
    io << l;

    auto l2 = NamedList::parse(io);
    // cerr << l2;
    EXPECT_EQ(l2, l);
  }

  { // scalar
    stringstream io;
    NamedList l("int", 1);
    io << l;

    auto l2 = NamedList::parse(io);
    EXPECT_EQ(l2, l);
  }

  { // list
    stringstream io;
    NamedList l;
    l.add("int", 1);
    l.add("double", 1.0);
    l.add("string", "str");

    l.add("bool", true);
    io << l;

    auto l2 = NamedList::parse(io);
    EXPECT_EQ(l2, l);
  }

  { // list with sublist
    stringstream io;
    NamedList l;
    l.add("int", 1);
    l.add("double", 1.0);
    l.add("string", "str");
    l.add("bool", false);
    l.add("recursive_sublist", l);
    l.add("empty_sublist", NamedList());

    io << l;
    auto l2 = NamedList::parse(io);

    EXPECT_EQ(l2, l);
  }

  { //deeply nested lists
    stringstream io;
    NamedList l;
    NamedList empty;
    NamedList l2;
    l2.add("empty", empty);
    l.add("l2", l2);
    l.add("int", 1);
    l.add("recursive", l);
    
    string content = l.to_string();
    NamedList res = NamedList::parse(content);

    EXPECT_EQ(l, res);
  }
}


TEST(NamedList, parse) {
  { 
    // basic
    string content = R"(
    {
      "Temperature": {
        "Cold": 10, 
        "Warm": 20, 
        "Hot":  30
      }
    }
    )";
    auto lst = NamedList::parse(content);
    NamedList ref;
    NamedList temp;
    temp.add("Cold", 10);
    temp.add("Warm", 20);
    temp.add("Hot", 30);
    ref.add("Temperature", temp);

    EXPECT_EQ(lst, ref);
  }

  { // single element
    string content = R"(
      {
        "Temperature": {
            "Cold": 10
        }
      }
      )";
    auto lst = NamedList::parse(content);
    NamedList ref;
    NamedList temp;
    temp.add("Cold", 10);
    ref.add("Temperature", temp);
    EXPECT_EQ(lst, ref);
  }

  {
  // empty line
    string content = R"(
      {
        "Temperature": {
          "Cold": 10
          #,"Warm": 20, 
          #,"Hot":  30
        }
      }
      )";

    cerr << stripComments(content);
    auto lst = NamedList::parse(stripComments(content));
    cerr << lst;
    NamedList ref;
    NamedList temp;
    temp.add("Cold", 10);
    ref.add("Temperature", temp);
    EXPECT_EQ(lst, ref);
  }

  { // regression: comma before
    string content = R"(
    "global_params":{
      "nb_rules":3
      ,"nb_max_var_per_rule":3
    }
    )";

    // cerr << stripComments(content);
    // logger().activate();
    auto lst = NamedList::parse(stripComments(content));
    // cerr << lst;
    NamedList ref("global_params");
    ref.add("nb_rules", 3);
    ref.add("nb_max_var_per_rule", 3);
    EXPECT_EQ(lst, ref);
  }
}

TEST(NamedList, parse_edge_cases) {
  {
    vector<string> contents = {"{}", "{ }", "{  }", "{ \n }", "{\n}"};
    for (auto content : contents) {
      auto lst = NamedList::parse(content);
      EXPECT_TRUE(lst.is_list());
      EXPECT_EQ(lst.size(), 0);
    }
  }

  {
    EXPECT_THROW(NamedList::parse("}"), runtime_error);
  }

  {
    EXPECT_EQ(NamedList::parse("1"), NamedList());
  }

  {
    // logger().activate();
    vector<string> contents = {
        R"({"name":-29})",
        R"({ "name":-29})",
        R"({"name":-29 })",
        R"({ "name" : -29 })",
        R"(     {"name":-29}      )"
    };
    for (auto content : contents) {
      auto lst = NamedList::parse(content);
      cerr << lst;
      NamedList ref;
      ref.add("name", -29);
      EXPECT_EQ(lst, ref);
    }
  }

  {
    // logger().activate();
    vector<string> contents = {
        R"({"name":-29})",
        R"({ "name":-29})",
        R"({"name":-29 })",
        R"({ "name" : -29 })",
        R"(     {"name":-29}      )"
    };
    for (auto content : contents) {
      auto lst = NamedList::parse(content);
      cerr << lst;
      NamedList ref;
      ref.add("name", -29);
      EXPECT_EQ(lst, ref);
    }
  }


}


TEST(NamedList, as_numeric_vector) {

  string content = R"(
{
  "Temperature": {
     "Cold": 10, 
     "Warm": 20.1, 
     "Hot":  30
  }
}
)";

  auto lst = NamedList::parse(content);

  EXPECT_THROW(lst.as_numeric_vector(), runtime_error);
  auto lst2 = lst.get_list("Temperature");

  vector<double> v = lst2.as_numeric_vector();
  EXPECT_EQ(v, vector<double>({10, 20.1, 30}));

  // ====== scalar case ====== 
  NamedList lst3;
  lst3.add("Temperature", 0.5);
  auto v2 = lst3.get_list("Temperature").as_numeric_vector();
  EXPECT_EQ(v2, vector<double>({0.5}));


  // =============== edge cases ===============
  { // empty
    NamedList lst;
    EXPECT_TRUE(lst.as_numeric_vector().empty());
  }

  { // not numeric
    NamedList lst;
    lst.add("key", "value");
    EXPECT_THROW(lst.as_numeric_vector(), runtime_error);
  }

}


TEST(NamedList, find_first_idx) {

  string content = R"(
{
  "name": {
     "item1": 10, 
     "item2": 20.1, 
     "item1":  "toto"
  }
}
)";
  auto lst = NamedList::parse(content);

  EXPECT_EQ(lst.find_first_idx("name"), 0);
  EXPECT_EQ(lst[0].find_first_idx("name"), -1);
  EXPECT_EQ(lst[0].find_first_idx("item1"), 0);
  EXPECT_EQ(lst[0].find_first_idx("item2"), 1);
  EXPECT_EQ(lst[0].find_first_idx("toto"), -1);

  EXPECT_EQ(lst["name"]["item1"].get_int(), 10);
  EXPECT_EQ(lst["name"]["item2"].get_double(), 20.1);

  EXPECT_THROW(lst["name"]["toto"], runtime_error);
}

TEST(NamedList, add_numeric_vector) {

  NamedList lst;
  vector<double> v = {0.1, 0.2, 0.3};
  lst.add("v", v);
  cerr << lst;

  string REF = R"({
    "v":{
      "1":0.1,
      "2":0.2,
      "3":0.3
    }
})";
  auto ref = NamedList::parse(REF);
  EXPECT_EQ(lst, ref);

  // edge cases
  NamedList lst2;
  lst2.add("v", vector<double>());
  
  EXPECT_EQ(lst2, NamedList::parse(R"({ "v":{} })"));
}

TEST(NamedList, add_map__as_string_numeric_map) {

  NamedList lst;

  map<string, double> hash;
  hash["feat1"] = 0;
  hash["feat2"] = 0.5;
  hash["feat3"] = 1;

  lst.add("hash", hash);
  cerr << lst;

  string REF = R"(
{
  "hash":{
    "feat1":0.0,
    "feat2":0.5,
    "feat3":1.0
  }
})";
  EXPECT_EQ(lst, NamedList::parse(REF));

  auto hash2 = lst["hash"].as_string_numeric_map();
  EXPECT_EQ(hash2, hash);
  // cerr << hash2;

  { // empty
    NamedList lst;
    EXPECT_TRUE(lst.as_string_numeric_map().empty());
  }

  { // not numeric
    NamedList lst;
    lst.add("key", "value");
    EXPECT_THROW(lst.as_string_numeric_map(), runtime_error);
  }


}

