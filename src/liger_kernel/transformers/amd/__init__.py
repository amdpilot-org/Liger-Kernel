"""AMD ROCm (gfx9*) vendor-specific transformer ops for Liger-Kernel.

Submodules here are activated via ``get_device_arch()`` dispatch in
``liger_kernel.transformers.layer_norm`` (and sibling wrappers), so that
gfx950-aware Triton launch configurations are used on MI355X instead of the
generic SM80/90-tuned defaults.
"""
