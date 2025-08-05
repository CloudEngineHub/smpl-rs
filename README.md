<div align="center">

# 🚶‍♂️ SMPL-rs

**Smpl-rs is the suite of SMPL functionality implemented in Rust over [gloss](https://github.com/Meshcapade/gloss). It has features for creating smpl-bodies, modifying and rendering them**

[![Crates.io](https://img.shields.io/crates/v/smpl-rs.svg)](https://crates.io/crates/smpl-rs)
[![PyPI](https://img.shields.io/pypi/v/smpl-rs.svg)](https://pypi.org/project/smpl-rs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Meshcapade/smpl-rs/LICENSE)

<img alt="SMPL-rs Banner" src="https://raw.githubusercontent.com/Meshcapade/smpl-rs/main/imgs/banner.png">

</div>

## Features 
- Run forward passes through SMPL models (betas -> mesh)
- Modify betas and expression parameters of the SMPL model in real time  
- Interfaces with [gloss](https://github.com/Meshcapade/gloss) for rendering meshes both in native and web
<div align="center">
<p align="middle">
  <img src="https://raw.githubusercontent.com/Meshcapade/smpl-rs/main/imgs/smpl.png" width="700"/>
</p>
</div>

## Documentation 
* [smpl-rs Rust API Documentation](https://docs.rs/smpl-rs/latest/smpl_rs/): Automatically generated docs for smpl-rs's Rust API
* [smpl-rs Rust Examples](https://github.com/Meshcapade/smpl-rs/tree/main/examples): smpl-rs's runnable examples in Rust, covering basic usage. 
* [smpl-rs Python Examples](https://github.com/Meshcapade/smpl-rs/tree/main/bindings/smpl_py/examples): smpl-rs's runnable examples for the Python bindings. Covers a wide range of features of the Python bindings. 

## Getting Started 
The easiest way to get started with smpl-rs is to install the Python bindings. 

```sh
$ pip install smpl-rs
```
Some examples of how to use the python bindings can be found in the python examples linked above.

## Data 
To use smpl-rs you need to download the SMPL-X data. 

* Download the models from [here](https://smpl-x.is.tue.mpg.de/download.php) (Download SMPL-X with removed headbun NPZ). 
* After this change the paths in the `misc_scripts/standardize_smpl.py` file to the path where you downloaded the models and where you want to save the standardized models. You will need some additional files provided in the `data/smplx` folder. 
* Then run as `python misc_scripts/standardize_smpl.py` to standardize the models. Lazy loading will need to be set to the path where you saved the standardized models. 

## Installation and Dependencies 
The main dependency is [gloss](https://github.com/Meshcapade/gloss) which will be downloaded and compiled automatically when building this package. You will need rust, and the rest is handled by cargo. To install Rust, simply run the following in your terminal: 

```sh
$ curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh
```

Some additional dependencies for Linux:

```sh
$ sudo apt-get install libvulkan-dev vulkan-tools xorg-dev libxkbcommon-x11-dev
```

For `MacOs`, it should run out of the box.


### For running the Rust examples
```sh
$ cd smpl-rs
$ cargo run --bin smpl_minimal
```

### For running the Python examples
```sh
$ cd smpl-rs/bindings/smpl_py
$ pip install gloss-rs smpl-rs 
$ ./examples/minimal.py
```

<!-- ### Build a
First install necessary dependencies
```sh
$ sudo apt install nodejs npm 
```

```sh
$ cd smpl-rs/examples/web/visualizer
$ wasm-pack build --target web
$ npm i
```
To run the web example we can create a dummy web server by opening another terminal and running:
```sh
$ cd smpl-rs/examples/web/visualizer
$ npm run start
# $ python -m http.server 
```
<!-- Finally navigate to `http://0.0.0.0:8000/smpl_webpage/` in your browser of choice. -->
<!-- Finally navigate to `http://localhost:3000/` in your browser of choice. --> 

<!-- ## Examples

Various examples can be found in the ./examples folder.\
You can run each one of them using 
```sh
$ cargo run --bin <example_name>
```

## React

Please read the file `examples/web/visualizer/README.md`  -->

## Some more Information
- The SMPL suite renders using [gloss](https://github.com/Meshcapade/gloss) and therefore uses an Entity-Component-System (ECS) framework. For more info on ECS check [here](https://bevyengine.org/learn/book/getting-started/ecs/). However to be noted that we use [Hecs] for our ECS system but most of them are very similar.
- Components like Animation and Betas are added to entities and that dictates which systems it uses. If you don't want animation on the avatar, just comment out the component for it when creating the entity. 
- For adding new functionality to [gloss](https://github.com/Meshcapade/gloss) we use callbacks. This is needed because on WASM the rendering loop cannot be explictly controlled.  