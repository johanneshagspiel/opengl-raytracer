<img src=resources/logo/opengl_raytracer_logo.png alt="OpenGL Raytracer" width="507" height="216">

--------------------------------------------------------------------------------
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/johanneshagspiel/opengl-raytracer/LICENSE.md)
[![Top Language](https://img.shields.io/github/languages/top/johanneshagspiel/opengl-raytracer)](https://github.com/johanneshagspiel/opengl-raytracer)
[![Latest Release](https://img.shields.io/github/v/release/johanneshagspiel/opengl-raytracer)](https://github.com/johanneshagspiel/opengl-raytracer/releases/)

# OpenGL Raytracer

This repository contains a full raytracer that was created in C++ and uses [Tucano](https://www.lcg.ufrj.br/tucano/) as a OpenGL wrapper to create from a virtual scene a static image. 

## Features

This full raytracer:

- performs ray intersections with planes, triangles, and bounding boxes.
- computates shading at the first impact point (diffuse and specular).
- performs recursive raytracing for re ections to simulate specular materials.
- calculates hard shadows from a point light.
- calculates soft shadows from a spherical light centered at a point light.
- has a simple acceleration structure.

## Tools

| Purpose                | Name                                                                                                            |
|------------------------|-----------------------------------------------------------------------------------------------------------------|
| Programming language   | C++                                                                                           |
| OpenGL library         | [Tucano](https://www.lcg.ufrj.br/tucano/)                                                                                                      |

## Installation Process

It is assumed that the users is using Windows. 

- Download and install [Visual Studio](https://visualstudio.microsoft.com/).
- When asked which other packages to install, make sure to tick the following options:
  - C++ CLI Support
  - Windows 10 SDK for Desktop C++ Development
  - VC++ 2015.3 (v140) toolset for C++ Development or a newer Version
- Load the repository as a folder in "Visual Studio" and then double-click on "raytracing.sln"
- Click on "Start without debugging" or press "Crtl + F5" to run the program

## Contributors

This raytracer was created together with:

- [Marko Matušovič](https://github.com/MMarko333)
- [Emre Ozkan](https://github.com/emre6943)
- [Omar Sheasha](https://github.com/osheasha)
- [Stef Rasing](https://github.com/stefstef00)
- [Lukas Zim](https://github.com/LukasZim)

## Licence

This "OpenGL Raytracer" is published under the MIT licence, which can be found in the [LICENSE](LICENSE) file. For this repository, the terms laid out there shall not apply to any individual that is currently enrolled at a higher education institution as a student. Those individuals shall not interact with any other part of this repository besides this README in any way by, for example cloning it or looking at its source code or have someone else interact with this repository in any way.

## References

The base image for the logo was taken from the [official Tucano website](https://www.lcg.ufrj.br/tucano/tucano.png). 
