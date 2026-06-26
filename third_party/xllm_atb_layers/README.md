# xllm_atb_layers (Source Dependency)

This directory is built as part of the top-level `xllm` project.
It is not an independent installable package in the current workflow.

## Integration Model

- Source path: `third_party/xllm_atb_layers`
- Integrated by: `xllm/CMakeLists.txt` via `add_subdirectory(...)`
- Library target: `xllm_atb_layers` (`cc_library` in this directory)

## Build Outputs

Artifacts are staged under the project build directory:

- `build/xllm_atb_layers/lib/libxllm_atb_layers.a`
- `build/xllm_atb_layers/include/xllm_atb_layers` (symlink to source tree)
