OpenSfM ![Docker workflow](https://github.com/mapillary/opensfm/workflows/Docker%20CI/badge.svg)
=======

## Overview
OpenSfM is a Structure from Motion library written in Python. The library serves as a processing pipeline for reconstructing camera poses and 3D scenes from multiple images. It consists of basic modules for Structure from Motion (feature detection/matching, minimal solvers) with a focus on building a robust and scalable reconstruction pipeline. It also integrates external sensor (e.g. GPS, accelerometer) measurements for geographical alignment and robustness. A JavaScript viewer is provided to preview the models and debug the pipeline.

<p align="center">
  <img src="https://opensfm.org/docs/_images/berlin_viewer.jpg" />
</p>

Checkout this [blog post with more demos](http://blog.mapillary.com/update/2014/12/15/sfm-preview.html)


## Getting Started

* [Building the library][]
  * first install [opencv](http://opencv.org/) & [ceres solver](http://ceres-solver.org/)
    * http://ceres-solver.org/installation.html
    * http://ceres-solver.org/ceres-solver-2.0.0.tar.gz
    * `cd ceres-solver; mkdir build; cd build`
    * `cmake ..; make -j3; sudo make install`
  * `git clone --recursive https://github.com/mapillary/OpenSfM`
  * `cd OpenSfM; pip3 install -r requirements.txt --user; python3 setup.py build`
    * https://github.com/mapillary/OpenSfM/blob/main/Dockerfile
* [Running a reconstruction][]
  * `bin/opensfm_run_all data/berlin`
  * `./viewer/node_modules.sh`
  * `python3 viewer/server.py -d data/berlin`
* [Documentation][]

[Building the library]: https://opensfm.org/docs/building.html "OpenSfM building instructions"
[Running a reconstruction]: https://opensfm.org/docs/using.html "OpenSfM usage"
[Documentation]: https://opensfm.org/docs/ "OpenSfM documentation"

## License

OpenSfM is BSD-style licensed, as found in the LICENSE file.  See also the Facebook Open Source [Terms of Use][] and [Privacy Policy][]

[Terms of Use]: https://opensource.facebook.com/legal/terms "Facebook Open Source - Terms of Use"
[Privacy Policy]: https://opensource.facebook.com/legal/privacy "Facebook Open Source - Privacy Policy"
