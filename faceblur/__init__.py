"""faceblur: high-precision face blurring for egocentric robot video on NVIDIA GPUs.

Kept import-light on purpose: the CPU encoder processes import only `faceblur.splice`.
Use `from faceblur.pipeline import Recipe, run` for the GPU pipeline.
"""
__version__ = "0.2.0"
