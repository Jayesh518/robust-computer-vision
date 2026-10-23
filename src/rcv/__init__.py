"""rcv: tools for the "Robust Computer Vision Under Difficult Conditions" project.

Modules
  labels   label space and conversion from dataset annotations to the 6 project classes
  corrupt  deterministic synthetic degradations (low light, fog, rain, snow, motion blur, noise)
  stats    mean/std and bootstrap confidence intervals
  eval     thin wrappers around Ultralytics validation and latency measurement (needs torch)
"""

__version__ = "0.1.0"
