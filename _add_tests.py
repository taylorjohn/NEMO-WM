# Append new tests to test_neuro_vlm_gate.py
new_tests = '''

class TestTextGoalDA:
    """Tests for the two-DA channel: text-goal similarity blending."""

    def test_update_text_goal_hot(self):
        neuro = BiologicalNeuromodulator()
        da = neuro.update_text_goal(0.26)  # above HOT=0.245
        assert da == 1.0

    def test_update_text_goal_cold(self):
        neuro = BiologicalNeuromodulator()
        da = neuro.update_text_goal(0.15)  # below COLD=0.180
        assert da == 0.0

    def test_update_text_goal_midrange(self):
        neuro = BiologicalNeuromodulator()
        da = neuro.update_text_goal(0.21)  # dead zone
        assert 0.0 < da < 1.0

    def test_text_goal_da_property(self):
        neuro = BiologicalNeuromodulator()
        neuro.update_text_goal(0.26)
        assert neuro.text_goal_da == 1.0

    def test_blended_da_weights(self):
        neuro = BiologicalNeuromodulator()
        neuro._state.da = 0.5
        neuro.update_text_goal(0.26)  # text_goal_da=1.0
        # expect 0.7*0.5 + 0.3*1.0 = 0.65
        assert abs(neuro.blended_da - 0.65) < 0.01

    def test_blended_da_temporal_only(self):
        neuro = BiologicalNeuromodulator()
        neuro._state.da = 0.8
        neuro.update_text_goal(0.15)  # text_goal_da=0.0
        # expect 0.7*0.8 + 0.3*0.0 = 0.56
        assert abs(neuro.blended_da - 0.56) < 0.01

    def test_default_text_goal_da_neutral(self):
        neuro = BiologicalNeuromodulator()
        assert neuro.text_goal_da == 0.5  # neutral start

    def test_set_text_goal_on_gated_vlm(self):
        import torch, torch.nn as nn
        vision = nn.Sequential(nn.Flatten(), nn.Linear(12288, 128))
        neuro  = BiologicalNeuromodulator()
        gated  = NeurallyGatedVLM(vision, neuro)
        proj   = torch.randn(128)
        gated.set_text_goal(proj)
        assert hasattr(gated, '_text_goal_proj')
        assert gated._text_goal_proj is not None

    def test_clear_text_goal(self):
        import torch, torch.nn as nn
        vision = nn.Sequential(nn.Flatten(), nn.Linear(12288, 128))
        neuro  = BiologicalNeuromodulator()
        gated  = NeurallyGatedVLM(vision, neuro)
        gated.set_text_goal(torch.randn(128))
        gated.clear_text_goal()
        assert gated._text_goal_proj is None
'''

src = open('test_neuro_vlm_gate.py', encoding='utf-8').read()
if 'TestTextGoalDA' not in src:
    open('test_neuro_vlm_gate.py', 'a', encoding='utf-8').write(new_tests)
    print('Tests added')
else:
    print('Already present')
