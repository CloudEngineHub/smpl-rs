# Changelog

All notable changes to this project will be documented in this file.

Please keep one empty line before and after all headers. (This is required for
`git` to produce a conflict when a release is made while a PR is open and the
PR's changelog entry would go into the wrong section).

And please only add new entries to the top of this list, right below the `#
Unreleased` header.

# Unreleased

# 0.9.0

### ⭐ Added
- Added SmplXS support
- Added vertex offsets as part of the SMPL forward pass with Scene GUI
- Added slerp interpolation for poses and exposed to python
- Added support for torch backend

### 🔧 Changed
- 90 degree rotation about up axis now happens across camera local up for UE5.7

### 🐛 Fixed
- Fixed quaternion to `axis_angle` conversion and interpolation jitter
- Fixed codec assuming mcs files always had `SceneAnimations`


# 0.8.0

### ⭐ Added
- Implemented animated props
- Added `VertexOffset` component with GUI for modifying offset strength
- Added Mcs support for exporting, parsing from gloss-scene, and `mcs2metadata` binary
- Python bindings can now export glTFs with bodies for specified indices

### 🔧 Changed
- Pose and Expression now use Burn tensors with multibackend support
- Faster `apply_pose` for smplx on GPU backends like WGPU
- Refactored glTF export to support body only, prop only, or both
- Updated dependencies of burn, wgpu, and egui


# 0.7.0

### ⭐ Added
- Added ARKit regressor and `ARKitModel` with smplx-arkit blendshapes
- `FaceType` parameter for specifying which blendshapes to use for the face and glTF export
- Added ONNX burn features and python scripts for converting models to ONNX

### 🔧 Changed
- Rescaling for expression blend weights/shapes for Blender in Smpl compat mode

### 🐛 Fixed
- Fixed smplx reading when the smplx has less betas than what is requested


# 0.6.0

<!-- ### ⚠️ BREAKING -->
<!-- ### ⭐ Added -->
### 🔧 Changed
- Updated Gloss to v0.6.0
- Some changes to make the glTF more conformant and more options for export 
- `.mcs` Scene features exposed to python bindings 
- Successive Entities now have different colors according to a fixed palette 


# 0.5.0

<!-- ### ⚠️ BREAKING -->
<!-- ### ⭐ Added -->
### 🔧 Changed
- Updated Gloss to v0.5.0
- Performance improvements by calculating normals and tangents only once when passing the smpl mesh onto gloss. 
- Pinned the rust version of stable and nightly so the CI doesn't randomly break 
<!-- ### 🐛 Fixed -->



# 0.4.0

### 🔧 Changed
- Updated Gloss to v0.4.0


# 0.3.0

<!-- ### ⚠️ BREAKING -->
<!-- ### ⭐ Added -->
### 🔧 Changed
- Made `Var` for measurements also derive `EnumString` so we can convert between string and the `Var`.
<!-- ### 🐛 Fixed -->


# 0.2.0

### ⭐ Added
- Added pose correctives for the GLTF export 

### 🔧 Changed
- Pose interpolation just clamps when weights is outside the [0,1] range instead of panicking.


# 0.1.3

### ⚠️ BREAKING
- Renamed `idx_vuv_2_vnouv` to `idx_split_2_merged`

### ⭐ Added
- Added GLTF export with animations 
- Added flag for Follower to specify if we follow with the camera, lights or both

<!-- ### 🔧 Changed -->
<!-- ### 🐛 Fixed -->


# 0.1.2

### ⚠️ BREAKING
- The forward pass of the smpl model now returns a `SmplOutput` instead of 3 matrices.


### ⭐ Added
- Exposed more structures likes the `SmplOutput` to python
- Added example of showing skeleton of smpl


### 🔧 Changed
- Updated to Gloss 0.1.2
- Made most of the internal arrays of the smpl model to be behind an arc so getting the model from the `SmplModels` is now just a shallow copy.
<!-- ### 🐛 Fixed -->

# 0.1.1

-Updated to Gloss 0.1.1

# 0.1.0

- Initial version on private pypi