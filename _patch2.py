import re

src = open('neuro_vlm_gate.py', encoding='utf-8').read()

# Find BiologicalNeuromodulator class and add after __init__
# Insert update_text_goal, text_goal_da property, blended_da property

NEW_METHODS = '''
    def update_text_goal(self, sim: float) -> float:
        """Update DA from text-goal visual similarity (calibrated April 2026).
        HOT=0.245, COLD=0.180, alpha=0.30"""
        self._text_goal_sim = float(sim)
        if sim >= 0.245:
            self._text_goal_da = 1.0
        elif sim <= 0.180:
            self._text_goal_da = 0.0
        else:
            self._text_goal_da = (sim - 0.180) / (0.245 - 0.180)
        return self._text_goal_da

    @property
    def text_goal_da(self) -> float:
        return getattr(self, '_text_goal_da', 0.5)

    @property
    def blended_da(self) -> float:
        """0.7*temporal_DA + 0.3*text_goal_DA"""
        return 0.70 * float(self._state.da) + 0.30 * self.text_goal_da

'''

# Insert before get_attention_gains or first non-init method
for target in ['    def get_attention_gains', '    def update_from_error',
               '    def is_fatigued', '    def rest']:
    idx = src.find(target)
    if idx > 0:
        src = src[:idx] + NEW_METHODS + src[idx:]
        print(f'Inserted before {target.strip()}')
        break

# Add set_text_goal to NeurallyGatedVLM
SET_GOAL = '''
    def set_text_goal(self, text_proj) -> None:
        """Set projected CLIP text goal (128-D unit vector)."""
        import torch.nn.functional as _F
        self._text_goal_proj = _F.normalize(text_proj.detach().float(), dim=-1)

    def clear_text_goal(self) -> None:
        self._text_goal_proj = None

'''

idx_encode = src.find('    def encode(')
if idx_encode > 0:
    src = src[:idx_encode] + SET_GOAL + src[idx_encode:]
    print('set_text_goal added before encode()')

open('neuro_vlm_gate.py', 'w', encoding='utf-8').write(src)
compile(open('neuro_vlm_gate.py').read(), 'neuro_vlm_gate.py', 'exec')
print('Syntax OK')
print('Done')
