#include <filesystem>
#include "fuzzy_coco.h"
#include "logging_logger.h"
#include "file_utils.h"

using namespace fuzzy_coco;
using namespace logging;

struct Params
{
  string datasetFile;
  string paramsFile;
  string fuzzyFile;
  string ouputPath;

  bool verbose = false;
  bool eval = false;
  bool predict = false;
  int seed = -1;
  int nb_output_vars = 1;
};

/**
 * Displays the command line help.
 */
void showHelp()
{
  string help = R"(
Valid parameters are :

 --evaluate   : Perform an evaluation of the given fuzzy system on the specified database
 --predict    : Perform a prediction of the given fuzzy system on the specified database
 --verbose    : Verbose output
 --seed value : seed for the random generator
 --nbout nb   : number of output variables (defaults to 1)

 -d path : Dataset  (REQUIRED)
 -p path : JSON parameters file (REQUIRED for fuzzy system inference)
 -f path : Fuzzy system  (REQUIRED for evaluation/prediction)
 -o path : Output Path (defaults to stdout)
 --help : this message
)";
  cerr << help;
}

/**
 * Prompts the invalid usage error message.
 *
 * @param progName Name of this executable.
 */
void invalidUsage(string progName)
{
  cerr << endl
       << "ERROR : Invalid parameters format !" << endl
       << endl;
  cerr << "Usage : " << progName << " -p1 value -p2 value ..." << endl
       << endl;
  cerr << "For parameter list run with --help" << endl
       << endl;
}

/**
 * Prompts the invalid parameter error message.
 */
void invalidParam(string param)
{
  cerr << endl
       << "ERROR : Invalid parameter '" << param << "'!" << endl;
  showHelp();
  exit(1);
}

void missingParamValue(string param)
{
  cerr << endl
       << "ERROR : parameter '" << param << " requires a value'!" << endl;
  showHelp();
  exit(1);
}

void error(const string &message)
{
  cerr << endl
       << "ERROR: " << message << endl
       << endl;
  showHelp();
  exit(1);
}

/**
 * Parse the command line arguments.
 *
 * @param args Arguments passed in a StringList structure.
 */
Params parseArguments(const vector<string> &args)
{
  Params params;
  // Look if we run directly from cmdline
  if (args.size() <= 1) return params;
  if (args.at(1) == "--help") showHelp();

  int i = 1;
  const int nb = args.size();
  while (i < nb)
  {
    // Look for the argument marker : '-'
    const string &arg = args.at(i);

    if (arg.at(0) != '-')
      invalidUsage(args[0]);
    if (arg.length() < 2)
      invalidParam(arg);

    // options that do not take values
    if (arg == "--verbose") {
      params.verbose = true;
    } else if (arg == "--evaluate") {
      params.eval = true;
    } else if (arg == "--predict") {
      params.predict = true;
    } else { // from there  we need a value
      if (i >= nb - 1)
        missingParamValue(arg);

      if (arg == "--nbout") {
        params.nb_output_vars = stoi(args.at(i + 1));
      } else if (arg.at(1) == 'd')  {
        params.datasetFile = args[i + 1];
      } else if (arg.at(1) == 'p') {
        params.paramsFile = args[i + 1];
      }  else if (arg.at(1) == 'f') {
        params.fuzzyFile = args[i + 1];
      } else if (arg.at(1) == 'o') {
        params.ouputPath = args[i + 1];
      } else if (arg == "--seed") {
        params.seed = stoi(args.at(i + 1));
      } else {
        invalidParam(arg);
      }
      i++; // value param
    }
    i++; // param
  } // while

  return params;
}

void check_file(const string &filename)
{
  if (filename.empty())
    return;
  if (!filesystem::is_regular_file(filename)) {
    cerr << "ERROR : file '" << filename << "' not found !" << endl
         << endl;
    error("file not found");
  }
}

void check_params(const Params &params)
{
  if (params.eval || params.predict) {
    if (params.eval && params.predict)
      error("you cannot perform both a prediction and a evaluation !");
    if (params.fuzzyFile.empty())
      error("you must specify a fuzzy system to perform a evaluation/prediction !");
    if (params.datasetFile.empty())
      error("you must specify a dataset to perform a evaluation/prediction !");
  }
  else {
    if (params.datasetFile.empty() || params.paramsFile.empty())  {
      error("you must load a dataset AND a params file to compute a FuzzySystem");
    }
  }

  check_file(params.datasetFile);
  check_file(params.paramsFile);
  check_file(params.fuzzyFile);
}



void launch(const Params &params)
{
  // read dataset
  DataFrame df = DataFrame::load(params.datasetFile, true);
  if (params.predict) {
    auto predicted = FuzzyCoco::loadAndPredict(df, params.fuzzyFile);
    FileUtils::writeCSV(cout, predicted);
    return;
  }
  if (params.eval) {
    FuzzyCoco::evalAndSave(df, params.fuzzyFile, cout);
    return;
  }

  auto input_params = NamedList::parse(StringUtils::stripComments(FileUtils::slurp(params.paramsFile)));
  FuzzyCocoParams coco_params(input_params);
  // cerr << StringUtils::stripComments(FileUtils::slurp(params.paramsFile));
  // cerr << input_params;

  logger() << L_time << "FuzzyCoco::searchBestFuzzySystem()...\n";
  auto results = FuzzyCoco::searchBestFuzzySystem(df, params.nb_output_vars, coco_params, params.seed);

  if (results.empty()) {
    cerr << "No results found\n";
    return;
  }

  if (params.ouputPath.empty()) {
    cout << results;
  } else {
    logger() << L_time << "saving fuzzy system in " << params.ouputPath << endl;
    FileUtils::mkdir_if_needed(params.ouputPath);
    fstream output_file(params.ouputPath, ios::out);
    if (!output_file.is_open()) throw runtime_error(string("unable to open file ") + params.ouputPath);
    output_file << results;
  }
}

/**
 * Main function.
 */

int main(int argc, char *argv[])
{
  vector<string> args;
  args.reserve(argc);
  for (int i = 0; i < argc; i++)
    args.push_back({argv[i]});

  Params params = parseArguments(args);
  check_params(params);

  if (params.verbose)
    logger().activate();
  logger() << L_allwaysFlush << L_time << "Fuzzy Coco started\n";

  launch(params);
}
