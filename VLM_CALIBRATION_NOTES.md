
VLM Gate Calibration Finding (April 11 2026):
==============================================
The neuro_vlm_gate.py uses DA for prediction error (existing):
  da_threshold = 0.0613 (DINOv2), 0.1424 (CLIP)
  Triggers REOBSERVE when z_t deviates from z_{t-1}

Text-visual similarity (new measurement):
  Outdoor RECON frames vs outdoor text: mean=0.21, range=0.10
  Indoor text vs outdoor frames: mean=0.14 (-0.07 gap)
  Calibrated thresholds: HOT=0.245, COLD=0.180

Architecture gap:
  Text goal proximity needs a SEPARATE DA channel from prediction error.
  Current: DA = f(temporal_error)
  Needed:  DA = alpha*f(temporal_error) + (1-alpha)*g(text_similarity)
  
Next: add text_goal_da parameter to BiologicalNeuromodulator.__init__
      and blend in NeurallyGatedVLM.encode()
