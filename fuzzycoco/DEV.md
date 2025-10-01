fuzzy coco: developers documentation
====================================


## building / testing 

### building the dev/debug version

`make build` will build the dev (not optimized version) suitable for development, testing and test coverage.

All generated files will be in `.build/`, the `fuzzycoco.exe` executable in 
`.build/src/` and the tests executables in `.build/tests/unit`

### running the unit tests

- `make test` will run all the tests.
- `make build test` will recompile what's needed and run all the tests
- `make verbose` will run all the tests and show the output
- `make debug` will show you details about the tests that failed if any
- to run a particular test, you can do something like `make build && .build/tests/unit/fuzzy_system_test`


### running the end-to-end tests

TODO

### test coverage

- `make build coverage` will generate a HTML test coverage report in `.coverage/src/index.html`.

