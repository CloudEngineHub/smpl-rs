# Changelog

All notable changes to this project will be documented in this file.

Please keep one empty line before and after all headers. (This is required for
`git` to produce a conflict when a release is made while a PR is open and the
PR's changelog entry would go into the wrong section).

And please only add new entries to the top of this list, right below the `#
Unreleased` header.

# Unreleased

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