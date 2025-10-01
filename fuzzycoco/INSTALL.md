INSTALLATION
=============

## requirements

This requires 

- a C++ 2017 compliant compiler.
- make and cmake
- lcov for the test coverage
- gdb and valgrind can be very useful for debugging

## devcontainer

A vscode (Visual Studio Code) devcontainer is provided, cf the `.devcontainer/` folder.
You can build and execute the software inside since it provides all required software. 
You could also just use the docker image (cf `.devcontainer/Dockerfile`) directly.

## building

### building the Release version

This the optimized version.

```
cd Release
make
```
This builds in in `Release/.build/.bin/`: 
  - `fuzzycoco.exe`: the executable
  - `libfuzzycoco.so`: the C++ shared library
  - `libfuzzycoco_static.a`: the C++ static library

and copies them in `bin/`.

### building the dev/debug version

[From the project root folder, ] `make build` will build the dev (not optimized version) suitable for development, testing and test coverage.

All generated files will be in `.build/`, the `fuzzycoco.exe` executable in 
`.build/src/` and the tests executables in `.build/tests/unit`

## quick testing

just run `make quick-test`
It is a shortcut for running all tests in `tests/e2e`