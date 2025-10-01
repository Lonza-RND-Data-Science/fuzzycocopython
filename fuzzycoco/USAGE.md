How to use fuzzycoco <!-- omit from toc -->
====================
- [Synopsis](#synopsis)
- [Overview](#overview)
- [Fuzzy System Inference (or fit)](#fuzzy-system-inference-or-fit)
- [Fuzzy System evaluation](#fuzzy-system-evaluation)
- [Fuzzy System prediction](#fuzzy-system-prediction)
- [file formats](#file-formats)
  - [`params.json`](#paramsjson)
  - [data file (input, output, both)](#data-file-input-output-both)
  - [`fuzzy_system.json`](#fuzzy_systemjson)
    - [Fuzzy system parameters](#fuzzy-system-parameters)
  - [Script](#script)

N.B: we will focus here on using the `fuzzycoco.exe` executable.

## Synopsis

```
# fit
fuzzycoco.exe -d DATA.csv -p PARAMS.json --seed 123 > fuzzy_system.json
# evaluate
fuzzycoco.exe -d OTHER_DATA.csv -f fuzzy_system.json  --evaluate > eval.json
# predict
fuzzycoco.exe -d INPUT.csv -f fuzzy_system.json --predict > outcome.csv
```

## Overview

There are three use cases:

  1.  **Fuzzy System Inference (or fit)**. the goal is to start from a dataset of observations contains **input** and **output** variables and given a 
      set of parameters to a **Fuzzy System** (Fuzzy rules + Membership functions): `fuzzy_system=fit(input, output, params)`

  1. **Outcome Prediction**. Given an existing *Fuzzy System*, and a dataset of input variables, predict the corresponding **output** variables: `output=predict(fuzzy_system, input)`

  1.**Fuzzy System evaluation**. Given an existing *Fuzzy System*, and a dataset of input and output variables, evaluate
    the fitness of the fuzzy system on that dataset. `fitnesss=evaluate(fuzzy_system, input, output)`


## Fuzzy System Inference (or fit)

To fit a fuzzy system, you need a dataset `DATA.csv` and a set of params `PARAMS.json`. 
This will generate a fuzzy system (cf [file formats](#file-formats)).
It is highly suggested to set a RNG seed using `--seed` for reproducibility.


```
# output on stdout
fuzzycoco.exe -d DATA.csv -p PARAMS.json --seed 123 | less
# output in a file
fuzzycoco.exe -d DATA.csv -p PARAMS.json --seed 123 -o results.json
# verbose
fuzzycoco.exe -d DATA.csv -p PARAMS.json --seed 123 --verbose
```


## Fuzzy System evaluation

The goal is to evaluate the performance of a given fuzzy system `fuzzy_system.json` on a given dataset `OTHER_DATA.csv`:

```
fuzzycoco.exe -d OTHER_DATA.csv -f fuzzy_system.json  --evaluate > eval.json
```

This will generate a JSON output with the computed fitness along with fitness metrics.

## Fuzzy System prediction

The goal is to predict an outcome given a fuzzy system `fuzzy_system.json` on some input data `INPUT.csv`:

Note that fuzzycoco.exe is smart enough, so that you can give it as input a full dataset (INPUT + OUTPUT variables)
and it will automatically extract the INPUT data, so that you do not need to extract it yourself.

```
fuzzycoco.exe -d INPUT.csv -f fuzzy_system.json --predict > outcome.csv
# or: give it a full dataset
fuzzycoco.exe -d OTHER_DATA.csv -f fuzzy_system.json --predict > outcome.csv
```

The output is a CSV dataset of the predictions for each of the output variables.


## file formats

### `params.json`

The parameters file is in **JSON** with comments, i.e. you can add inline comments in the file with `#`.
Most parameters have a default value, some may be computed from other parameters. 
You will set some parameters in the `params.json` file, that will always have precedence over default values or computed
values.

Check [PARAMS.md](./PARAMS.md) for the parameters documentation.
You may also look at one example: [/tests/e2e/iris36/params.json](./tests/e2e/iris36/params.json)

### data file (input, output, both)

The data file must:

  - be in **CSV** format 
  - use a semi-colon `;` delimiter, 
  - have a header line
  - have output variable(s) appear AFTER the input variables.

### `fuzzy_system.json`

The description of a fuzzy system is a **JSON** file.
The overall structure is:

  - fit{}
    - fitness
    - metrics{}
    - generations
  - fuzzy_system{}
    - variables{}
      - input[]
      - output[]
    - rules[], 
    - default_rules[]
  - params{}

You can find an example here: [/tests/e2e/iris36/results/seed.123/fuzzy_system.json](./tests/e2e/iris36/results/seed.123/fuzzy_system.json)





It is also possible to choose the threshold applied to the output values. Generally, it is selected in the middle of the minimum and maximum value.


#### Fuzzy system parameters
These parameters determine the shape of the emerged fuzzy systems (number of rules, maximum number of variables per rule, number of input/output sets) and the number of bits used to code the different elements (variables, rules, membership functions) in the genome.

There is also parameter that indicates to the software how much output variables are present in the database. Without this indication, the application has no means to know the number output variables.

Be very careful when editing these parameters. Incoherent values can lead to erroneous results and even to the crash of the application.

### Script
The script files have the .fs extension and are coded in the Javascript language. They are composed of the following elements:
- The definition of all the parameters
- The definition of the doSetParams() function
- The definition of the doRun() function which controls the execution of multiple (or single) runs of coevolution

All these elements must be present and complete in the script in order to make it valid. A reference valid script file called ref.fs is present in the bin/script/ folder. Its syntax is quite simple and should be easily understood. Basically, after having defined all the parameters and implemented the doSetParams() function, the doRun() function defines how much runs of evolution will be performed and which parameters will be modified between the runs.

