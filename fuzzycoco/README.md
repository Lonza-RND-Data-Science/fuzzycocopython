fuzzycoco
====================================================================================================

[![unit tests](https://github.com/Lonza-RND-Data-Science/fuzzycoco/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/Lonza-RND-Data-Science/fuzzycoco/actions/workflows/unit_tests.yml)

[![codecov](https://codecov.io/github/Lonza-RND-Data-Science/fuzzycoco/graph/badge.svg?token=UMCPQXVXQA)](https://codecov.io/github/Lonza-RND-Data-Science/fuzzycoco)

## Introduction

This is a C++ implementation of the **FuzzyCoCo** algorithm.
It constructs systems that predict the outcome of a human decision-making 
process while providing an understandable explanation of a possible reasoning leading to it. 
The constructed fuzzy systems are composed of rules and linguistic variables. 
For more information about the method, please refer to Prof. Carlos Andrés Peña 
Reyes' thesis entitled "Coevolutionary Fuzzy Modeling" 
(*Fuzzy CoCo: a cooperative-coevolutionary approach to fuzzy modeling* from [Carlos Andrés Peña-Reyes](https://orcid.org/0000-0002-2113-6498)).

## License

This fuzzycoco software is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).  
See the [LICENSE](./LICENSE) file for details.

## Acknowledgements

This software is a complete rewrite of the FUGE-LC software by Karl Forner.
The goals of this reimplementation were:
  - to remove the dependency to the Qt C++ framework
  - to make it easily testable (redesign guided by the TDD (Test Driven Methodology))
  - to make it usable as a library, so that it could be wrapped in dynamic languages such as R and Python
  - to remove the previously mandatory javascript execution script

FUGE-LC credits:
- FUGE-LC by Jean-Philippe Meylan (2009-2010)
- Genetic algorithm update by Yvan Da Silva (2012)
- Code upgrade by [Rochus Keller](http://rochus-keller.ch/) <me@rochus.keller.ch> (2022)
- New feature: initialization using expert knowledge by Magali Egger<magali.egger@heig-vd.ch>, Data Scientist (2024)


## Contents

This repository provides:
  *  a C++ library
      * suitable for wrapping the software in dynamic languages (TODO: cf Rfuzzycoco)
      * well tested
  * an executable that used the library to demo its features. 
  * some tests, useful to show the input, parameters and output of the software.


## documentation

  - [INSTALL.md](./INSTALL.md): how to build and install the software
  - [USAGE.md](./USAGE.md): how to run the software
  - [PARAMS.md](./PARAMS.md): the description of the algorithm parameters 
  - [DEV.md](./DEV.md): instructions for developers and library users
  - [CHANGELOG.md](./CHANGELOG.md): All notable changes

