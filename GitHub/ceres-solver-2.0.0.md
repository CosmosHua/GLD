[![Build Status](https://travis-ci.org/ceres-solver/ceres-solver.svg?branch=master)](https://travis-ci.org/ceres-solver/ceres-solver)

Ceres Solver
============

Ceres Solver is an open source C++ library for modeling and solving
large, complicated optimization problems. It is a feature rich, mature
and performant library which has been used in production at Google
since 2010. Ceres Solver can solve two kinds of problems.

1. Non-linear Least Squares problems with bounds constraints.
2. General unconstrained optimization problems.

Please see [ceres-solver.org](http://ceres-solver.org/) for more information.

http://ceres-solver.org/ceres-solver-2.0.0.tar.gz

http://ceres-solver.org/installation.html

```bash
mkdir build; cd build
cmake ..; make -j3; #make test
# Optionally install Ceres, it can also be exported using CMake which allows Ceres to be used without requiring installation, see the documentation for the EXPORT_BUILD_DIR option for more information.
sudo make install
```

