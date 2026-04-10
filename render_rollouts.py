import gymnasium as gym
import gym_pusht
import numpy as np
import imageio
from pathlib import Path

Path("figures").mkdir(exist_ok=True)
env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array")
frames = []

for ep in range(3):
    obs, _ = env.reset(seed=ep*7)
    for step in range(150):
        frame = env.render()
        frames.append(frame)
        agent = obs[:2]; block = obs[2:4]; goal = np.array([0.65, 0.65])
        action = block if np.linalg.norm(agent-block)>0.15 else agent+(goal-block)*0.3
        obs, _, term, trunc, _ = env.step(np.clip(action,0,1))
        if term or trunc: break

env.close()
imageio.mimsave("figures/pusht_rollout.mp4", frames, fps=10, macro_block_size=1)
print(f"Saved {len(frames)} frames to figures/pusht_rollout.mp4")

(ryzen-ai-1.7.0) PS C:\Users\MeteorAI\Desktop\CORTEX> python hybrid_predictor.py --demo --flat-ckpt checkpoints/action_wm/action_wm_pusht_full_best.pt

Hybrid predictor demo (synthetic PushT)...
  Flat model loaded: ep=45 ac_lift=+0.6536
  Graph ckpt not found: checkpoints/graph_wm/graph_wm_pusht_best.pt — graph mode unavailable
  step 00  DA=0.000  mode=flat   k_ctx=12  dist=0.424
  step 01  DA=0.000  mode=flat   k_ctx=12  dist=0.210
  step 02  DA=1.000  mode=graph  k_ctx=12  dist=0.114
  step 03  DA=1.000  mode=graph  k_ctx=12  dist=0.025
  step 04  DA=0.850  mode=graph  k_ctx=12  dist=0.172
  step 05  DA=1.000  mode=graph  k_ctx=12  dist=0.095
  step 06  DA=0.850  mode=graph  k_ctx=12  dist=0.269
  step 07  DA=0.722  mode=graph  k_ctx=12  dist=0.160
  step 08  DA=1.000  mode=graph  k_ctx=12  dist=0.103
  step 09  DA=0.850  mode=graph  k_ctx=12  dist=0.292
  step 10  DA=0.722  mode=graph  k_ctx=12  dist=0.165
  step 11  DA=1.000  mode=graph  k_ctx=12  dist=0.115
  step 12  DA=0.850  mode=graph  k_ctx=12  dist=0.324
  step 13  DA=0.722  mode=graph  k_ctx=12  dist=0.194
  step 14  DA=1.000  mode=graph  k_ctx=12  dist=0.124
  step 15  DA=1.000  mode=graph  k_ctx=12  dist=0.064
  step 16  DA=0.850  mode=graph  k_ctx=12  dist=0.236
  step 17  DA=1.000  mode=graph  k_ctx=12  dist=0.134
  step 18  DA=1.000  mode=graph  k_ctx=12  dist=0.089
  step 19  DA=0.850  mode=graph  k_ctx=12  dist=0.333
(ryzen-ai-1.7.0) PS C:\Users\MeteorAI\Desktop\CORTEX>


[ep00 s00350] L=2.8486 L_pred=2.7444 L_idm=0.0021 ac_lift=+0.0510 gate=2.077 lr=4.47e-04
[ep00 s00400] L=2.8457 L_pred=2.7328 L_idm=0.0023 ac_lift=+0.0587 gate=2.302 lr=5.50e-04

Epoch 00  loss=4.4663  ac_lift=+0.0574  idm=0.0269
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  → Saved: checkpoints\graph_wm\graph_wm_pusht_best.pt
[ep01 s00450] L=2.7733 L_pred=2.6656 L_idm=0.0022 ac_lift=+0.0609 gate=2.486 lr=6.58e-04
[ep01 s00500] L=2.7176 L_pred=2.6224 L_idm=0.0019 ac_lift=+0.0571 gate=2.658 lr=7.70e-04
[ep01 s00550] L=2.7432 L_pred=2.6569 L_idm=0.0017 ac_lift=+0.0638 gate=2.854 lr=8.82e-04
[ep01 s00600] L=2.6605 L_pred=2.5540 L_idm=0.0021 ac_lift=+0.0640 gate=3.084 lr=9.91e-04
[ep01 s00650] L=2.5749 L_pred=2.4803 L_idm=0.0019 ac_lift=+0.0554 gate=3.289 lr=1.10e-03
[ep01 s00700] L=2.5630 L_pred=2.4773 L_idm=0.0017 ac_lift=+0.0510 gate=3.497 lr=1.19e-03
[ep01 s00750] L=2.5636 L_pred=2.4704 L_idm=0.0019 ac_lift=+0.0750 gate=3.766 lr=1.28e-03
[ep01 s00800] L=2.5172 L_pred=2.4309 L_idm=0.0017 ac_lift=+0.0540 gate=3.924 lr=1.35e-03

Epoch 01  loss=2.6597  ac_lift=+0.0607  idm=0.0020
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  → Saved: checkpoints\graph_wm\graph_wm_pusht_best.pt
[ep02 s00850] L=2.3644 L_pred=2.2698 L_idm=0.0019 ac_lift=+0.0563 gate=4.123 lr=1.41e-03
[ep02 s00900] L=2.5237 L_pred=2.4266 L_idm=0.0019 ac_lift=+0.0693 gate=4.333 lr=1.46e-03
[ep02 s00950] L=2.3883 L_pred=2.3037 L_idm=0.0017 ac_lift=+0.0624 gate=4.563 lr=1.49e-03
[ep02 s01000] L=2.3377 L_pred=2.2110 L_idm=0.0025 ac_lift=+0.0587 gate=4.724 lr=1.50e-03
[ep02 s01050] L=2.2151 L_pred=2.1280 L_idm=0.0017 ac_lift=+0.0779 gate=4.983 lr=1.50e-03
[ep02 s01100] L=2.2305 L_pred=2.1330 L_idm=0.0020 ac_lift=+0.0781 gate=5.214 lr=1.50e-03
[ep02 s01150] L=2.7751 L_pred=2.3586 L_idm=0.0083 ac_lift=+0.0693 gate=5.386 lr=1.50e-03
[ep02 s01200] L=2.2355 L_pred=2.0703 L_idm=0.0033 ac_lift=+0.0963 gate=5.755 lr=1.50e-03

Epoch 02  loss=2.4060  ac_lift=+0.0695  idm=0.0028
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  → Saved: checkpoints\graph_wm\graph_wm_pusht_best.pt
[ep03 s01250] L=2.2284 L_pred=2.0687 L_idm=0.0032 ac_lift=+0.0603 gate=5.856 lr=1.50e-03
[ep03 s01300] L=2.4680 L_pred=2.2091 L_idm=0.0052 ac_lift=+0.0611 gate=6.010 lr=1.50e-03
[ep03 s01350] L=2.1562 L_pred=2.0621 L_idm=0.0019 ac_lift=+0.1007 gate=6.261 lr=1.50e-03
[ep03 s01400] L=2.0972 L_pred=2.0324 L_idm=0.0013 ac_lift=+0.0996 gate=6.375 lr=1.50e-03
[ep03 s01450] L=2.1175 L_pred=2.0441 L_idm=0.0015 ac_lift=+0.0922 gate=6.490 lr=1.50e-03
[ep03 s01500] L=2.1049 L_pred=2.0299 L_idm=0.0015 ac_lift=+0.1443 gate=6.624 lr=1.50e-03
[ep03 s01550] L=2.1834 L_pred=2.0948 L_idm=0.0018 ac_lift=+0.1156 gate=6.661 lr=1.50e-03
[ep03 s01600] L=2.0693 L_pred=1.9822 L_idm=0.0017 ac_lift=+0.1226 gate=6.776 lr=1.50e-03

Epoch 03  loss=2.1525  ac_lift=+0.0975  idm=0.0021
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  ✓ Block edge dominates goal edge (contact-aware routing)
  → Saved: checkpoints\graph_wm\graph_wm_pusht_best.pt
[ep04 s01650] L=1.9471 L_pred=1.8751 L_idm=0.0014 ac_lift=+0.1340 gate=6.787 lr=1.50e-03
[ep04 s01700] L=2.0041 L_pred=1.9156 L_idm=0.0018 ac_lift=+0.0996 gate=6.851 lr=1.50e-03
[ep04 s01750] L=2.0144 L_pred=1.9564 L_idm=0.0012 ac_lift=+0.1454 gate=6.969 lr=1.49e-03
[ep04 s01800] L=1.9550 L_pred=1.8748 L_idm=0.0016 ac_lift=+0.1465 gate=7.054 lr=1.49e-03
[ep04 s01850] L=1.9864 L_pred=1.9262 L_idm=0.0012 ac_lift=+0.1413 gate=7.176 lr=1.49e-03
[ep04 s01900] L=1.9085 L_pred=1.8232 L_idm=0.0017 ac_lift=+0.0948 gate=7.099 lr=1.49e-03
[ep04 s01950] L=2.0311 L_pred=1.9805 L_idm=0.0010 ac_lift=+0.1387 gate=7.208 lr=1.49e-03
[ep04 s02000] L=1.9124 L_pred=1.8532 L_idm=0.0012 ac_lift=+0.1347 gate=7.298 lr=1.49e-03

Epoch 04  loss=2.0110  ac_lift=+0.1317  idm=0.0014
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  → Saved: checkpoints\graph_wm\graph_wm_pusht_best.pt
[ep05 s02050] L=1.9176 L_pred=1.8545 L_idm=0.0013 ac_lift=+0.0911 gate=7.267 lr=1.49e-03
[ep05 s02100] L=1.8889 L_pred=1.8291 L_idm=0.0012 ac_lift=+0.1155 gate=7.392 lr=1.49e-03
[ep05 s02150] L=1.9331 L_pred=1.8838 L_idm=0.0010 ac_lift=+0.1414 gate=7.474 lr=1.49e-03
[ep05 s02200] L=2.0126 L_pred=1.9597 L_idm=0.0011 ac_lift=+0.1507 gate=7.510 lr=1.49e-03
[ep05 s02250] L=1.9160 L_pred=1.8637 L_idm=0.0010 ac_lift=+0.1544 gate=7.574 lr=1.48e-03
[ep05 s02300] L=1.8913 L_pred=1.8370 L_idm=0.0011 ac_lift=+0.1604 gate=7.618 lr=1.48e-03
[ep05 s02350] L=1.8339 L_pred=1.7830 L_idm=0.0010 ac_lift=+0.1374 gate=7.650 lr=1.48e-03
[ep05 s02400] L=1.8925 L_pred=1.8384 L_idm=0.0011 ac_lift=+0.1489 gate=7.719 lr=1.48e-03

Epoch 05  loss=1.9162  ac_lift=+0.1395  idm=0.0012
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  → Saved: checkpoints\graph_wm\graph_wm_pusht_best.pt
[ep06 s02450] L=1.9844 L_pred=1.9249 L_idm=0.0012 ac_lift=+0.1557 gate=7.768 lr=1.48e-03
[ep06 s02500] L=1.8625 L_pred=1.8141 L_idm=0.0010 ac_lift=+0.1422 gate=7.804 lr=1.48e-03
[ep06 s02550] L=1.9051 L_pred=1.8545 L_idm=0.0010 ac_lift=+0.1543 gate=7.842 lr=1.48e-03
[ep06 s02600] L=1.7912 L_pred=1.7384 L_idm=0.0011 ac_lift=+0.1229 gate=7.867 lr=1.47e-03
[ep06 s02650] L=1.8275 L_pred=1.7733 L_idm=0.0011 ac_lift=+0.1347 gate=7.896 lr=1.47e-03
[ep06 s02700] L=1.8665 L_pred=1.7983 L_idm=0.0014 ac_lift=+0.1549 gate=7.936 lr=1.47e-03
[ep06 s02750] L=1.8277 L_pred=1.7770 L_idm=0.0010 ac_lift=+0.1492 gate=7.962 lr=1.47e-03
[ep06 s02800] L=1.8701 L_pred=1.8173 L_idm=0.0011 ac_lift=+0.1416 gate=7.989 lr=1.47e-03

Epoch 06  loss=1.8929  ac_lift=+0.1462  idm=0.0011
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  ✓ Block edge dominates goal edge (contact-aware routing)
  → Saved: checkpoints\graph_wm\graph_wm_pusht_best.pt
[ep07 s02850] L=1.8345 L_pred=1.7764 L_idm=0.0012 ac_lift=+0.1530 gate=8.029 lr=1.47e-03
[ep07 s02900] L=1.8707 L_pred=1.8254 L_idm=0.0009 ac_lift=+0.1476 gate=8.053 lr=1.46e-03
[ep07 s02950] L=1.8344 L_pred=1.7810 L_idm=0.0011 ac_lift=+0.1560 gate=8.083 lr=1.46e-03
[ep07 s03000] L=1.8368 L_pred=1.7801 L_idm=0.0011 ac_lift=+0.1594 gate=8.145 lr=1.46e-03
[ep07 s03050] L=1.7805 L_pred=1.7361 L_idm=0.0009 ac_lift=+0.1524 gate=8.173 lr=1.46e-03
[ep07 s03100] L=1.8136 L_pred=1.7574 L_idm=0.0011 ac_lift=+0.1573 gate=8.181 lr=1.46e-03
[ep07 s03150] L=1.8683 L_pred=1.8166 L_idm=0.0010 ac_lift=+0.1479 gate=8.254 lr=1.45e-03
[ep07 s03200] L=1.7963 L_pred=1.7348 L_idm=0.0012 ac_lift=+0.1595 gate=8.269 lr=1.45e-03

Epoch 07  loss=1.8344  ac_lift=+0.1527  idm=0.0011
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  → Saved: checkpoints\graph_wm\graph_wm_pusht_best.pt
[ep08 s03250] L=1.8170 L_pred=1.7691 L_idm=0.0010 ac_lift=+0.1596 gate=8.320 lr=1.45e-03
[ep08 s03300] L=1.8851 L_pred=1.8326 L_idm=0.0011 ac_lift=+0.1717 gate=8.366 lr=1.45e-03
[ep08 s03350] L=1.7724 L_pred=1.7207 L_idm=0.0010 ac_lift=+0.1824 gate=8.377 lr=1.45e-03
[ep08 s03400] L=1.8106 L_pred=1.7599 L_idm=0.0010 ac_lift=+0.1818 gate=8.405 lr=1.44e-03
[ep08 s03450] L=1.7889 L_pred=1.7457 L_idm=0.0009 ac_lift=+0.1698 gate=8.465 lr=1.44e-03
[ep08 s03500] L=1.7546 L_pred=1.7127 L_idm=0.0008 ac_lift=+0.1562 gate=8.496 lr=1.44e-03
[ep08 s03550] L=1.7723 L_pred=1.7263 L_idm=0.0009 ac_lift=+0.1782 gate=8.541 lr=1.44e-03
[ep08 s03600] L=1.6980 L_pred=1.6489 L_idm=0.0010 ac_lift=+0.1780 gate=8.559 lr=1.43e-03

Epoch 08  loss=1.7925  ac_lift=+0.1698  idm=0.0010
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  → Saved: checkpoints\graph_wm\graph_wm_pusht_best.pt
[ep09 s03650] L=1.7815 L_pred=1.7246 L_idm=0.0011 ac_lift=+0.1687 gate=8.573 lr=1.43e-03
[ep09 s03700] L=1.7433 L_pred=1.6896 L_idm=0.0011 ac_lift=+0.1804 gate=8.627 lr=1.43e-03
[ep09 s03750] L=1.7553 L_pred=1.7100 L_idm=0.0009 ac_lift=+0.1772 gate=8.673 lr=1.43e-03
[ep09 s03800] L=1.7118 L_pred=1.6640 L_idm=0.0010 ac_lift=+0.1831 gate=8.713 lr=1.42e-03
[ep09 s03850] L=1.7902 L_pred=1.7392 L_idm=0.0010 ac_lift=+0.1744 gate=8.743 lr=1.42e-03
[ep09 s03900] L=1.6721 L_pred=1.6215 L_idm=0.0010 ac_lift=+0.1697 gate=8.773 lr=1.42e-03
[ep09 s03950] L=1.8169 L_pred=1.7641 L_idm=0.0011 ac_lift=+0.1847 gate=8.821 lr=1.41e-03
[ep09 s04000] L=1.7333 L_pred=1.6840 L_idm=0.0010 ac_lift=+0.1972 gate=8.850 lr=1.41e-03

Epoch 09  loss=1.7557  ac_lift=+0.1858  idm=0.0010
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  → Saved: checkpoints\graph_wm\graph_wm_pusht_best.pt
[ep10 s04050] L=1.7303 L_pred=1.6791 L_idm=0.0010 ac_lift=+0.1952 gate=8.884 lr=1.41e-03
[ep10 s04100] L=1.7449 L_pred=1.7024 L_idm=0.0008 ac_lift=+0.2036 gate=8.906 lr=1.41e-03
[ep10 s04150] L=1.6874 L_pred=1.6406 L_idm=0.0009 ac_lift=+0.2015 gate=8.943 lr=1.40e-03
[ep10 s04200] L=1.6509 L_pred=1.6091 L_idm=0.0008 ac_lift=+0.1927 gate=8.994 lr=1.40e-03
[ep10 s04250] L=1.6518 L_pred=1.6069 L_idm=0.0009 ac_lift=+0.1875 gate=9.012 lr=1.40e-03
[ep10 s04300] L=1.6566 L_pred=1.6092 L_idm=0.0009 ac_lift=+0.1800 gate=9.047 lr=1.39e-03
[ep10 s04350] L=1.6915 L_pred=1.6394 L_idm=0.0010 ac_lift=+0.1786 gate=9.072 lr=1.39e-03
[ep10 s04400] L=1.6458 L_pred=1.5974 L_idm=0.0010 ac_lift=+0.1848 gate=9.090 lr=1.39e-03

Epoch 10  loss=1.6961  ac_lift=+0.1965  idm=0.0010
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  ✓ Block edge dominates goal edge (contact-aware routing)
  → Saved: checkpoints\graph_wm\graph_wm_pusht_best.pt
[ep11 s04450] L=1.6295 L_pred=1.5840 L_idm=0.0009 ac_lift=+0.1956 gate=9.149 lr=1.38e-03
[ep11 s04500] L=1.6622 L_pred=1.6206 L_idm=0.0008 ac_lift=+0.1836 gate=9.145 lr=1.38e-03
[ep11 s04550] L=1.6244 L_pred=1.5850 L_idm=0.0008 ac_lift=+0.2073 gate=9.188 lr=1.38e-03
[ep11 s04600] L=1.6422 L_pred=1.5928 L_idm=0.0010 ac_lift=+0.2027 gate=9.215 lr=1.37e-03
[ep11 s04650] L=1.6221 L_pred=1.5777 L_idm=0.0009 ac_lift=+0.1991 gate=9.231 lr=1.37e-03
[ep11 s04700] L=1.6082 L_pred=1.5675 L_idm=0.0008 ac_lift=+0.1956 gate=9.286 lr=1.37e-03
[ep11 s04750] L=1.6930 L_pred=1.6469 L_idm=0.0009 ac_lift=+0.1952 gate=9.327 lr=1.36e-03
[ep11 s04800] L=1.6988 L_pred=1.6532 L_idm=0.0009 ac_lift=+0.2114 gate=9.364 lr=1.36e-03

Epoch 11  loss=1.6471  ac_lift=+0.2018  idm=0.0009
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  ✓ Block edge dominates goal edge (contact-aware routing)
  → Saved: checkpoints\graph_wm\graph_wm_pusht_best.pt
[ep12 s04850] L=1.6098 L_pred=1.5675 L_idm=0.0008 ac_lift=+0.2288 gate=9.378 lr=1.36e-03
[ep12 s04900] L=1.6267 L_pred=1.5829 L_idm=0.0009 ac_lift=+0.1998 gate=9.400 lr=1.35e-03
[ep12 s04950] L=1.6680 L_pred=1.6237 L_idm=0.0009 ac_lift=+0.2011 gate=9.428 lr=1.35e-03
[ep12 s05000] L=1.6331 L_pred=1.5869 L_idm=0.0009 ac_lift=+0.1979 gate=9.430 lr=1.35e-03
[ep12 s05050] L=1.6095 L_pred=1.5626 L_idm=0.0009 ac_lift=+0.2009 gate=9.475 lr=1.34e-03
[ep12 s05100] L=1.6368 L_pred=1.5878 L_idm=0.0010 ac_lift=+0.2030 gate=9.501 lr=1.34e-03
[ep12 s05150] L=1.6071 L_pred=1.5683 L_idm=0.0008 ac_lift=+0.1958 gate=9.517 lr=1.33e-03
[ep12 s05200] L=1.5898 L_pred=1.5448 L_idm=0.0009 ac_lift=+0.1879 gate=9.523 lr=1.33e-03
[ep12 s05250] L=1.6234 L_pred=1.5775 L_idm=0.0009 ac_lift=+0.2057 gate=9.556 lr=1.33e-03

Epoch 12  loss=1.6273  ac_lift=+0.2036  idm=0.0009
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  → Saved: checkpoints\graph_wm\graph_wm_pusht_best.pt
[ep13 s05300] L=1.5990 L_pred=1.5574 L_idm=0.0008 ac_lift=+0.2049 gate=9.599 lr=1.32e-03
[ep13 s05350] L=1.6717 L_pred=1.6267 L_idm=0.0009 ac_lift=+0.1935 gate=9.625 lr=1.32e-03
[ep13 s05400] L=1.6469 L_pred=1.6018 L_idm=0.0009 ac_lift=+0.1946 gate=9.653 lr=1.31e-03
[ep13 s05450] L=1.6020 L_pred=1.5586 L_idm=0.0009 ac_lift=+0.2186 gate=9.675 lr=1.31e-03
[ep13 s05500] L=1.6586 L_pred=1.6078 L_idm=0.0010 ac_lift=+0.2265 gate=9.707 lr=1.31e-03
[ep13 s05550] L=1.6087 L_pred=1.5638 L_idm=0.0009 ac_lift=+0.2061 gate=9.745 lr=1.30e-03
[ep13 s05600] L=1.6176 L_pred=1.5710 L_idm=0.0009 ac_lift=+0.2352 gate=9.756 lr=1.30e-03
[ep13 s05650] L=1.5948 L_pred=1.5580 L_idm=0.0007 ac_lift=+0.2147 gate=9.780 lr=1.29e-03

Epoch 13  loss=1.6179  ac_lift=+0.2115  idm=0.0009
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  → Saved: checkpoints\graph_wm\graph_wm_pusht_best.pt
[ep14 s05700] L=1.5707 L_pred=1.5257 L_idm=0.0009 ac_lift=+0.2263 gate=9.816 lr=1.29e-03
[ep14 s05750] L=1.6471 L_pred=1.6072 L_idm=0.0008 ac_lift=+0.2250 gate=9.841 lr=1.29e-03
[ep14 s05800] L=1.6616 L_pred=1.5852 L_idm=0.0015 ac_lift=+0.1955 gate=9.851 lr=1.28e-03
[ep14 s05850] L=1.8039 L_pred=1.7480 L_idm=0.0011 ac_lift=+0.2050 gate=9.893 lr=1.28e-03
[ep14 s05900] L=1.6913 L_pred=1.6495 L_idm=0.0008 ac_lift=+0.2122 gate=9.926 lr=1.27e-03
[ep14 s05950] L=1.6276 L_pred=1.5792 L_idm=0.0010 ac_lift=+0.2257 gate=9.957 lr=1.27e-03
[ep14 s06000] L=1.6750 L_pred=1.6337 L_idm=0.0008 ac_lift=+0.2248 gate=9.954 lr=1.26e-03
[ep14 s06050] L=1.6646 L_pred=1.6258 L_idm=0.0008 ac_lift=+0.2349 gate=9.989 lr=1.26e-03

Epoch 14  loss=1.6560  ac_lift=+0.2206  idm=0.0009
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
[ep15 s06100] L=1.6790 L_pred=1.6344 L_idm=0.0009 ac_lift=+0.2259 gate=10.020 lr=1.25e-03
[ep15 s06150] L=1.6614 L_pred=1.6129 L_idm=0.0010 ac_lift=+0.2147 gate=10.022 lr=1.25e-03
[ep15 s06200] L=1.7140 L_pred=1.6718 L_idm=0.0008 ac_lift=+0.2339 gate=10.043 lr=1.25e-03
[ep15 s06250] L=1.6253 L_pred=1.5802 L_idm=0.0009 ac_lift=+0.2266 gate=10.057 lr=1.24e-03
[ep15 s06300] L=1.6219 L_pred=1.5772 L_idm=0.0009 ac_lift=+0.2162 gate=10.076 lr=1.24e-03
[ep15 s06350] L=1.5836 L_pred=1.5400 L_idm=0.0009 ac_lift=+0.2235 gate=10.095 lr=1.23e-03
[ep15 s06400] L=1.6946 L_pred=1.6482 L_idm=0.0009 ac_lift=+0.2378 gate=10.117 lr=1.23e-03
[ep15 s06450] L=1.5950 L_pred=1.5574 L_idm=0.0008 ac_lift=+0.2224 gate=10.144 lr=1.22e-03

Epoch 15  loss=1.6430  ac_lift=+0.2266  idm=0.0009
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
[ep16 s06500] L=1.6078 L_pred=1.5698 L_idm=0.0008 ac_lift=+0.2389 gate=10.146 lr=1.22e-03
[ep16 s06550] L=1.6161 L_pred=1.5740 L_idm=0.0008 ac_lift=+0.2360 gate=10.149 lr=1.21e-03
[ep16 s06600] L=1.6532 L_pred=1.6149 L_idm=0.0008 ac_lift=+0.2377 gate=10.171 lr=1.21e-03
[ep16 s06650] L=1.6190 L_pred=1.5776 L_idm=0.0008 ac_lift=+0.2284 gate=10.195 lr=1.20e-03
[ep16 s06700] L=1.5710 L_pred=1.5309 L_idm=0.0008 ac_lift=+0.2351 gate=10.207 lr=1.20e-03
[ep16 s06750] L=1.5652 L_pred=1.5197 L_idm=0.0009 ac_lift=+0.2313 gate=10.228 lr=1.19e-03
[ep16 s06800] L=1.6330 L_pred=1.5930 L_idm=0.0008 ac_lift=+0.2373 gate=10.230 lr=1.19e-03
[ep16 s06850] L=1.6044 L_pred=1.5547 L_idm=0.0010 ac_lift=+0.2573 gate=10.270 lr=1.18e-03

Epoch 16  loss=1.6042  ac_lift=+0.2316  idm=0.0008
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  → Saved: checkpoints\graph_wm\graph_wm_pusht_best.pt
[ep17 s06900] L=1.5998 L_pred=1.5606 L_idm=0.0008 ac_lift=+0.2380 gate=10.299 lr=1.18e-03
[ep17 s06950] L=1.5603 L_pred=1.5223 L_idm=0.0008 ac_lift=+0.2292 gate=10.319 lr=1.17e-03
[ep17 s07000] L=1.5792 L_pred=1.5296 L_idm=0.0010 ac_lift=+0.2302 gate=10.334 lr=1.17e-03
[ep17 s07050] L=1.6099 L_pred=1.5682 L_idm=0.0008 ac_lift=+0.2394 gate=10.334 lr=1.16e-03
[ep17 s07100] L=1.5241 L_pred=1.4851 L_idm=0.0008 ac_lift=+0.2515 gate=10.347 lr=1.16e-03
[ep17 s07150] L=1.5702 L_pred=1.5278 L_idm=0.0008 ac_lift=+0.2413 gate=10.370 lr=1.15e-03
[ep17 s07200] L=1.6233 L_pred=1.5762 L_idm=0.0009 ac_lift=+0.2353 gate=10.386 lr=1.15e-03
[ep17 s07250] L=1.5827 L_pred=1.5385 L_idm=0.0009 ac_lift=+0.2310 gate=10.401 lr=1.14e-03

Epoch 17  loss=1.5853  ac_lift=+0.2391  idm=0.0008
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  → Saved: checkpoints\graph_wm\graph_wm_pusht_best.pt
[ep18 s07300] L=1.6071 L_pred=1.5637 L_idm=0.0009 ac_lift=+0.2373 gate=10.415 lr=1.14e-03
[ep18 s07350] L=1.5834 L_pred=1.5440 L_idm=0.0008 ac_lift=+0.2336 gate=10.446 lr=1.13e-03
[ep18 s07400] L=1.5657 L_pred=1.5293 L_idm=0.0007 ac_lift=+0.2472 gate=10.458 lr=1.13e-03
[ep18 s07450] L=1.5236 L_pred=1.4755 L_idm=0.0010 ac_lift=+0.2507 gate=10.468 lr=1.12e-03
[ep18 s07500] L=1.6623 L_pred=1.6127 L_idm=0.0010 ac_lift=+0.2643 gate=10.488 lr=1.11e-03
[ep18 s07550] L=1.5327 L_pred=1.4907 L_idm=0.0008 ac_lift=+0.2534 gate=10.511 lr=1.11e-03
[ep18 s07600] L=1.5588 L_pred=1.5202 L_idm=0.0008 ac_lift=+0.2441 gate=10.536 lr=1.10e-03
[ep18 s07650] L=1.6093 L_pred=1.5707 L_idm=0.0008 ac_lift=+0.2477 gate=10.561 lr=1.10e-03

Epoch 18  loss=1.5858  ac_lift=+0.2460  idm=0.0008
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
[ep19 s07700] L=1.5974 L_pred=1.5546 L_idm=0.0009 ac_lift=+0.2552 gate=10.574 lr=1.09e-03
[ep19 s07750] L=1.5710 L_pred=1.5320 L_idm=0.0008 ac_lift=+0.2411 gate=10.584 lr=1.09e-03
[ep19 s07800] L=1.5969 L_pred=1.5574 L_idm=0.0008 ac_lift=+0.2586 gate=10.587 lr=1.08e-03
[ep19 s07850] L=1.5730 L_pred=1.5295 L_idm=0.0009 ac_lift=+0.2599 gate=10.613 lr=1.08e-03
[ep19 s07900] L=1.6220 L_pred=1.5796 L_idm=0.0008 ac_lift=+0.2577 gate=10.636 lr=1.07e-03
[ep19 s07950] L=1.6106 L_pred=1.5725 L_idm=0.0008 ac_lift=+0.2578 gate=10.652 lr=1.07e-03
[ep19 s08000] L=1.5528 L_pred=1.5096 L_idm=0.0009 ac_lift=+0.2362 gate=10.655 lr=1.06e-03
[ep19 s08050] L=1.5175 L_pred=1.4785 L_idm=0.0008 ac_lift=+0.2507 gate=10.675 lr=1.05e-03

Epoch 19  loss=1.5580  ac_lift=+0.2493  idm=0.0008
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  → Saved: checkpoints\graph_wm\graph_wm_pusht_best.pt
[ep20 s08100] L=1.5410 L_pred=1.4999 L_idm=0.0008 ac_lift=+0.2409 gate=10.690 lr=1.05e-03
[ep20 s08150] L=1.5616 L_pred=1.5251 L_idm=0.0007 ac_lift=+0.2481 gate=10.696 lr=1.04e-03
[ep20 s08200] L=1.5417 L_pred=1.5014 L_idm=0.0008 ac_lift=+0.2532 gate=10.711 lr=1.04e-03
[ep20 s08250] L=1.5925 L_pred=1.5523 L_idm=0.0008 ac_lift=+0.2636 gate=10.737 lr=1.03e-03
[ep20 s08300] L=1.5171 L_pred=1.4720 L_idm=0.0009 ac_lift=+0.2623 gate=10.740 lr=1.03e-03
[ep20 s08350] L=1.5096 L_pred=1.4700 L_idm=0.0008 ac_lift=+0.2521 gate=10.749 lr=1.02e-03
[ep20 s08400] L=1.5500 L_pred=1.5166 L_idm=0.0007 ac_lift=+0.2476 gate=10.768 lr=1.01e-03
[ep20 s08450] L=1.4997 L_pred=1.4612 L_idm=0.0008 ac_lift=+0.2402 gate=10.792 lr=1.01e-03

Epoch 20  loss=1.5305  ac_lift=+0.2492  idm=0.0008
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  → Saved: checkpoints\graph_wm\graph_wm_pusht_best.pt
[ep21 s08500] L=1.5143 L_pred=1.4705 L_idm=0.0009 ac_lift=+0.2627 gate=10.796 lr=1.00e-03
[ep21 s08550] L=1.5210 L_pred=1.4817 L_idm=0.0008 ac_lift=+0.2481 gate=10.804 lr=9.97e-04
[ep21 s08600] L=1.5664 L_pred=1.5259 L_idm=0.0008 ac_lift=+0.2563 gate=10.825 lr=9.92e-04
[ep21 s08650] L=1.4863 L_pred=1.4458 L_idm=0.0008 ac_lift=+0.2489 gate=10.831 lr=9.86e-04
[ep21 s08700] L=1.4994 L_pred=1.4602 L_idm=0.0008 ac_lift=+0.2439 gate=10.859 lr=9.80e-04
[ep21 s08750] L=1.5634 L_pred=1.5206 L_idm=0.0009 ac_lift=+0.2412 gate=10.881 lr=9.74e-04
[ep21 s08800] L=1.4865 L_pred=1.4467 L_idm=0.0008 ac_lift=+0.2669 gate=10.887 lr=9.68e-04
[ep21 s08850] L=1.5680 L_pred=1.5271 L_idm=0.0008 ac_lift=+0.2651 gate=10.891 lr=9.62e-04

Epoch 21  loss=1.5075  ac_lift=+0.2563  idm=0.0008
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  → Saved: checkpoints\graph_wm\graph_wm_pusht_best.pt
[ep22 s08900] L=1.5104 L_pred=1.4720 L_idm=0.0008 ac_lift=+0.2541 gate=10.915 lr=9.57e-04
[ep22 s08950] L=1.4927 L_pred=1.4574 L_idm=0.0007 ac_lift=+0.2535 gate=10.924 lr=9.51e-04
[ep22 s09000] L=1.5554 L_pred=1.5198 L_idm=0.0007 ac_lift=+0.2556 gate=10.929 lr=9.45e-04
[ep22 s09050] L=1.5523 L_pred=1.5084 L_idm=0.0009 ac_lift=+0.2608 gate=10.936 lr=9.39e-04
[ep22 s09100] L=1.5081 L_pred=1.4638 L_idm=0.0009 ac_lift=+0.2642 gate=10.946 lr=9.33e-04
[ep22 s09150] L=1.4970 L_pred=1.4597 L_idm=0.0007 ac_lift=+0.2524 gate=10.966 lr=9.27e-04
[ep22 s09200] L=1.5036 L_pred=1.4642 L_idm=0.0008 ac_lift=+0.2589 gate=10.976 lr=9.21e-04
[ep22 s09250] L=1.5605 L_pred=1.5223 L_idm=0.0008 ac_lift=+0.2515 gate=10.979 lr=9.15e-04

Epoch 22  loss=1.5103  ac_lift=+0.2567  idm=0.0008
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
[ep23 s09300] L=1.5463 L_pred=1.5073 L_idm=0.0008 ac_lift=+0.2536 gate=10.997 lr=9.09e-04
[ep23 s09350] L=1.4683 L_pred=1.4346 L_idm=0.0007 ac_lift=+0.2754 gate=10.999 lr=9.03e-04
[ep23 s09400] L=1.4488 L_pred=1.4132 L_idm=0.0007 ac_lift=+0.2646 gate=11.008 lr=8.97e-04
[ep23 s09450] L=1.5328 L_pred=1.4944 L_idm=0.0008 ac_lift=+0.2672 gate=11.012 lr=8.91e-04
[ep23 s09500] L=1.4863 L_pred=1.4455 L_idm=0.0008 ac_lift=+0.2494 gate=11.020 lr=8.85e-04
[ep23 s09550] L=1.4954 L_pred=1.4573 L_idm=0.0008 ac_lift=+0.2487 gate=11.030 lr=8.79e-04
[ep23 s09600] L=1.4734 L_pred=1.4334 L_idm=0.0008 ac_lift=+0.2246 gate=11.022 lr=8.73e-04
[ep23 s09650] L=1.5163 L_pred=1.4788 L_idm=0.0008 ac_lift=+0.2925 gate=11.055 lr=8.67e-04

Epoch 23  loss=1.4956  ac_lift=+0.2584  idm=0.0008
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  ✓ Block edge dominates goal edge (contact-aware routing)
  → Saved: checkpoints\graph_wm\graph_wm_pusht_best.pt
[ep24 s09700] L=1.4899 L_pred=1.4522 L_idm=0.0008 ac_lift=+0.2659 gate=11.072 lr=8.61e-04
[ep24 s09750] L=1.4485 L_pred=1.4094 L_idm=0.0008 ac_lift=+0.2510 gate=11.082 lr=8.55e-04
[ep24 s09800] L=1.5046 L_pred=1.4658 L_idm=0.0008 ac_lift=+0.2906 gate=11.092 lr=8.48e-04
[ep24 s09850] L=1.4698 L_pred=1.4356 L_idm=0.0007 ac_lift=+0.2544 gate=11.107 lr=8.42e-04
[ep24 s09900] L=1.4717 L_pred=1.4254 L_idm=0.0009 ac_lift=+0.2573 gate=11.112 lr=8.36e-04
[ep24 s09950] L=1.4750 L_pred=1.4374 L_idm=0.0008 ac_lift=+0.2768 gate=11.115 lr=8.30e-04
[ep24 s10000] L=1.4710 L_pred=1.4344 L_idm=0.0007 ac_lift=+0.2473 gate=11.123 lr=8.24e-04
[ep24 s10050] L=1.4675 L_pred=1.4253 L_idm=0.0008 ac_lift=+0.2493 gate=11.140 lr=8.18e-04
[ep24 s10100] L=1.5078 L_pred=1.4700 L_idm=0.0008 ac_lift=+0.2493 gate=11.144 lr=8.12e-04

Epoch 24  loss=1.4837  ac_lift=+0.2613  idm=0.0008
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  → Saved: checkpoints\graph_wm\graph_wm_pusht_best.pt
[ep25 s10150] L=1.4820 L_pred=1.4441 L_idm=0.0008 ac_lift=+0.2599 gate=11.152 lr=8.06e-04
[ep25 s10200] L=1.4649 L_pred=1.4320 L_idm=0.0007 ac_lift=+0.2621 gate=11.160 lr=8.00e-04
[ep25 s10250] L=1.4718 L_pred=1.4347 L_idm=0.0007 ac_lift=+0.2644 gate=11.171 lr=7.93e-04
[ep25 s10300] L=1.5306 L_pred=1.4950 L_idm=0.0007 ac_lift=+0.2605 gate=11.172 lr=7.87e-04
[ep25 s10350] L=1.5199 L_pred=1.4784 L_idm=0.0008 ac_lift=+0.2722 gate=11.179 lr=7.81e-04
[ep25 s10400] L=1.4869 L_pred=1.4511 L_idm=0.0007 ac_lift=+0.2470 gate=11.195 lr=7.75e-04
[ep25 s10450] L=1.4777 L_pred=1.4403 L_idm=0.0007 ac_lift=+0.2500 gate=11.204 lr=7.69e-04
[ep25 s10500] L=1.5132 L_pred=1.4741 L_idm=0.0008 ac_lift=+0.2635 gate=11.206 lr=7.63e-04

Epoch 25  loss=1.4813  ac_lift=+0.2606  idm=0.0008
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  → Saved: checkpoints\graph_wm\graph_wm_pusht_best.pt
[ep26 s10550] L=1.4749 L_pred=1.4315 L_idm=0.0009 ac_lift=+0.2637 gate=11.218 lr=7.57e-04
[ep26 s10600] L=1.4678 L_pred=1.4263 L_idm=0.0008 ac_lift=+0.2537 gate=11.228 lr=7.50e-04
[ep26 s10650] L=1.4420 L_pred=1.4043 L_idm=0.0008 ac_lift=+0.2532 gate=11.232 lr=7.44e-04
[ep26 s10700] L=1.4757 L_pred=1.4393 L_idm=0.0007 ac_lift=+0.2588 gate=11.241 lr=7.38e-04
[ep26 s10750] L=1.4633 L_pred=1.4275 L_idm=0.0007 ac_lift=+0.2627 gate=11.237 lr=7.32e-04
[ep26 s10800] L=1.5105 L_pred=1.4739 L_idm=0.0007 ac_lift=+0.2573 gate=11.244 lr=7.26e-04
[ep26 s10850] L=1.5035 L_pred=1.4575 L_idm=0.0009 ac_lift=+0.2572 gate=11.246 lr=7.20e-04
[ep26 s10900] L=1.4728 L_pred=1.4332 L_idm=0.0008 ac_lift=+0.2618 gate=11.249 lr=7.14e-04

Epoch 26  loss=1.4808  ac_lift=+0.2595  idm=0.0008
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  → Saved: checkpoints\graph_wm\graph_wm_pusht_best.pt
[ep27 s10950] L=1.5305 L_pred=1.4862 L_idm=0.0009 ac_lift=+0.2505 gate=11.258 lr=7.08e-04
[ep27 s11000] L=1.4514 L_pred=1.4161 L_idm=0.0007 ac_lift=+0.2547 gate=11.268 lr=7.01e-04
[ep27 s11050] L=1.4606 L_pred=1.4248 L_idm=0.0007 ac_lift=+0.2343 gate=11.268 lr=6.95e-04
[ep27 s11100] L=1.5049 L_pred=1.4635 L_idm=0.0008 ac_lift=+0.2581 gate=11.276 lr=6.89e-04
[ep27 s11150] L=1.4892 L_pred=1.4527 L_idm=0.0007 ac_lift=+0.2601 gate=11.280 lr=6.83e-04
[ep27 s11200] L=1.5155 L_pred=1.4762 L_idm=0.0008 ac_lift=+0.2622 gate=11.281 lr=6.77e-04
[ep27 s11250] L=1.4625 L_pred=1.4241 L_idm=0.0008 ac_lift=+0.2791 gate=11.285 lr=6.71e-04
[ep27 s11300] L=1.5109 L_pred=1.4713 L_idm=0.0008 ac_lift=+0.2643 gate=11.286 lr=6.65e-04

Epoch 27  loss=1.4789  ac_lift=+0.2600  idm=0.0008
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  → Saved: checkpoints\graph_wm\graph_wm_pusht_best.pt
[ep28 s11350] L=1.4601 L_pred=1.4207 L_idm=0.0008 ac_lift=+0.2549 gate=11.287 lr=6.59e-04
[ep28 s11400] L=1.4515 L_pred=1.4107 L_idm=0.0008 ac_lift=+0.2603 gate=11.295 lr=6.53e-04
[ep28 s11450] L=1.4442 L_pred=1.4106 L_idm=0.0007 ac_lift=+0.2689 gate=11.300 lr=6.46e-04
[ep28 s11500] L=1.4699 L_pred=1.4282 L_idm=0.0008 ac_lift=+0.2719 gate=11.309 lr=6.40e-04
[ep28 s11550] L=1.4571 L_pred=1.4186 L_idm=0.0008 ac_lift=+0.2611 gate=11.303 lr=6.34e-04
[ep28 s11600] L=1.4591 L_pred=1.4205 L_idm=0.0008 ac_lift=+0.2485 gate=11.309 lr=6.28e-04
[ep28 s11650] L=1.5060 L_pred=1.4687 L_idm=0.0007 ac_lift=+0.2561 gate=11.315 lr=6.22e-04
[ep28 s11700] L=1.4627 L_pred=1.4275 L_idm=0.0007 ac_lift=+0.2631 gate=11.314 lr=6.16e-04

Epoch 28  loss=1.4726  ac_lift=+0.2635  idm=0.0007
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  → Saved: checkpoints\graph_wm\graph_wm_pusht_best.pt
[ep29 s11750] L=1.4679 L_pred=1.4341 L_idm=0.0007 ac_lift=+0.2652 gate=11.316 lr=6.10e-04
[ep29 s11800] L=1.4993 L_pred=1.4623 L_idm=0.0007 ac_lift=+0.2698 gate=11.324 lr=6.04e-04
[ep29 s11850] L=1.4752 L_pred=1.4360 L_idm=0.0008 ac_lift=+0.2544 gate=11.323 lr=5.98e-04
[ep29 s11900] L=1.4802 L_pred=1.4407 L_idm=0.0008 ac_lift=+0.2675 gate=11.331 lr=5.92e-04
[ep29 s11950] L=1.4420 L_pred=1.4074 L_idm=0.0007 ac_lift=+0.2614 gate=11.332 lr=5.86e-04
[ep29 s12000] L=1.5437 L_pred=1.5092 L_idm=0.0007 ac_lift=+0.2586 gate=11.335 lr=5.80e-04
[ep29 s12050] L=1.4517 L_pred=1.4158 L_idm=0.0007 ac_lift=+0.2789 gate=11.339 lr=5.74e-04
[ep29 s12100] L=1.4874 L_pred=1.4550 L_idm=0.0006 ac_lift=+0.2549 gate=11.335 lr=5.68e-04

Epoch 29  loss=1.4721  ac_lift=+0.2649  idm=0.0007
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  → Saved: checkpoints\graph_wm\graph_wm_pusht_best.pt
[ep30 s12150] L=1.5158 L_pred=1.4792 L_idm=0.0007 ac_lift=+0.2574 gate=11.340 lr=5.62e-04
[ep30 s12200] L=1.4856 L_pred=1.4499 L_idm=0.0007 ac_lift=+0.2694 gate=11.345 lr=5.56e-04
[ep30 s12250] L=1.4887 L_pred=1.4529 L_idm=0.0007 ac_lift=+0.2629 gate=11.352 lr=5.50e-04
[ep30 s12300] L=1.4703 L_pred=1.4322 L_idm=0.0008 ac_lift=+0.2761 gate=11.354 lr=5.44e-04
[ep30 s12350] L=1.4655 L_pred=1.4295 L_idm=0.0007 ac_lift=+0.2563 gate=11.355 lr=5.39e-04
[ep30 s12400] L=1.4760 L_pred=1.4391 L_idm=0.0007 ac_lift=+0.2606 gate=11.353 lr=5.33e-04
[ep30 s12450] L=1.4844 L_pred=1.4472 L_idm=0.0007 ac_lift=+0.2534 gate=11.355 lr=5.27e-04
[ep30 s12500] L=1.5044 L_pred=1.4667 L_idm=0.0008 ac_lift=+0.2606 gate=11.358 lr=5.21e-04

Epoch 30  loss=1.4656  ac_lift=+0.2614  idm=0.0007
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  ✓ Block edge dominates goal edge (contact-aware routing)
  → Saved: checkpoints\graph_wm\graph_wm_pusht_best.pt
[ep31 s12550] L=1.4263 L_pred=1.3892 L_idm=0.0007 ac_lift=+0.2625 gate=11.369 lr=5.15e-04
[ep31 s12600] L=1.4984 L_pred=1.4624 L_idm=0.0007 ac_lift=+0.2754 gate=11.369 lr=5.09e-04
[ep31 s12650] L=1.4146 L_pred=1.3815 L_idm=0.0007 ac_lift=+0.2654 gate=11.366 lr=5.03e-04
[ep31 s12700] L=1.4952 L_pred=1.4573 L_idm=0.0008 ac_lift=+0.2562 gate=11.378 lr=4.98e-04
[ep31 s12750] L=1.4313 L_pred=1.3935 L_idm=0.0008 ac_lift=+0.2577 gate=11.377 lr=4.92e-04
[ep31 s12800] L=1.5272 L_pred=1.4935 L_idm=0.0007 ac_lift=+0.2613 gate=11.373 lr=4.86e-04
[ep31 s12850] L=1.5325 L_pred=1.4989 L_idm=0.0007 ac_lift=+0.2643 gate=11.380 lr=4.80e-04
[ep31 s12900] L=1.4776 L_pred=1.4384 L_idm=0.0008 ac_lift=+0.2773 gate=11.379 lr=4.75e-04

Epoch 31  loss=1.4717  ac_lift=+0.2653  idm=0.0007
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
[ep32 s12950] L=1.4276 L_pred=1.3921 L_idm=0.0007 ac_lift=+0.2477 gate=11.381 lr=4.69e-04
[ep32 s13000] L=1.4584 L_pred=1.4240 L_idm=0.0007 ac_lift=+0.2555 gate=11.379 lr=4.63e-04
[ep32 s13050] L=1.4707 L_pred=1.4327 L_idm=0.0008 ac_lift=+0.2781 gate=11.386 lr=4.58e-04
[ep32 s13100] L=1.4196 L_pred=1.3873 L_idm=0.0006 ac_lift=+0.2530 gate=11.392 lr=4.52e-04
[ep32 s13150] L=1.4676 L_pred=1.4334 L_idm=0.0007 ac_lift=+0.2699 gate=11.390 lr=4.46e-04
[ep32 s13200] L=1.5012 L_pred=1.4683 L_idm=0.0007 ac_lift=+0.2708 gate=11.401 lr=4.41e-04
[ep32 s13250] L=1.4571 L_pred=1.4242 L_idm=0.0007 ac_lift=+0.2754 gate=11.394 lr=4.35e-04
[ep32 s13300] L=1.5000 L_pred=1.4651 L_idm=0.0007 ac_lift=+0.2589 gate=11.389 lr=4.30e-04

Epoch 32  loss=1.4664  ac_lift=+0.2657  idm=0.0007
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
[ep33 s13350] L=1.4464 L_pred=1.4116 L_idm=0.0007 ac_lift=+0.2628 gate=11.392 lr=4.24e-04
[ep33 s13400] L=1.4309 L_pred=1.3922 L_idm=0.0008 ac_lift=+0.2700 gate=11.399 lr=4.19e-04
[ep33 s13450] L=1.4491 L_pred=1.4124 L_idm=0.0007 ac_lift=+0.2734 gate=11.401 lr=4.13e-04
[ep33 s13500] L=1.4761 L_pred=1.4395 L_idm=0.0007 ac_lift=+0.2737 gate=11.399 lr=4.08e-04
[ep33 s13550] L=1.4400 L_pred=1.4031 L_idm=0.0007 ac_lift=+0.2669 gate=11.406 lr=4.02e-04
[ep33 s13600] L=1.4660 L_pred=1.4339 L_idm=0.0006 ac_lift=+0.2624 gate=11.406 lr=3.97e-04
[ep33 s13650] L=1.4619 L_pred=1.4259 L_idm=0.0007 ac_lift=+0.2605 gate=11.407 lr=3.91e-04
[ep33 s13700] L=1.4303 L_pred=1.3984 L_idm=0.0006 ac_lift=+0.2513 gate=11.409 lr=3.86e-04

Epoch 33  loss=1.4628  ac_lift=+0.2649  idm=0.0007
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  → Saved: checkpoints\graph_wm\graph_wm_pusht_best.pt
[ep34 s13750] L=1.4546 L_pred=1.4252 L_idm=0.0006 ac_lift=+0.2573 gate=11.413 lr=3.81e-04
[ep34 s13800] L=1.4673 L_pred=1.4356 L_idm=0.0006 ac_lift=+0.2812 gate=11.416 lr=3.75e-04
[ep34 s13850] L=1.4795 L_pred=1.4499 L_idm=0.0006 ac_lift=+0.2587 gate=11.413 lr=3.70e-04
[ep34 s13900] L=1.4489 L_pred=1.4112 L_idm=0.0008 ac_lift=+0.2556 gate=11.410 lr=3.65e-04
[ep34 s13950] L=1.4383 L_pred=1.4060 L_idm=0.0006 ac_lift=+0.2666 gate=11.413 lr=3.59e-04
[ep34 s14000] L=1.4428 L_pred=1.4102 L_idm=0.0007 ac_lift=+0.2728 gate=11.414 lr=3.54e-04
[ep34 s14050] L=1.4872 L_pred=1.4565 L_idm=0.0006 ac_lift=+0.2692 gate=11.419 lr=3.49e-04
[ep34 s14100] L=1.4947 L_pred=1.4562 L_idm=0.0008 ac_lift=+0.2652 gate=11.415 lr=3.44e-04

Epoch 34  loss=1.4616  ac_lift=+0.2652  idm=0.0007
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  → Saved: checkpoints\graph_wm\graph_wm_pusht_best.pt
[ep35 s14150] L=1.4683 L_pred=1.4362 L_idm=0.0006 ac_lift=+0.2651 gate=11.416 lr=3.39e-04
[ep35 s14200] L=1.4468 L_pred=1.4101 L_idm=0.0007 ac_lift=+0.2872 gate=11.418 lr=3.34e-04
[ep35 s14250] L=1.4844 L_pred=1.4510 L_idm=0.0007 ac_lift=+0.2681 gate=11.418 lr=3.28e-04
[ep35 s14300] L=1.4521 L_pred=1.4136 L_idm=0.0008 ac_lift=+0.2716 gate=11.414 lr=3.23e-04
[ep35 s14350] L=1.4505 L_pred=1.4130 L_idm=0.0007 ac_lift=+0.2731 gate=11.417 lr=3.18e-04
[ep35 s14400] L=1.4880 L_pred=1.4553 L_idm=0.0007 ac_lift=+0.2586 gate=11.419 lr=3.13e-04
[ep35 s14450] L=1.4815 L_pred=1.4442 L_idm=0.0007 ac_lift=+0.2530 gate=11.419 lr=3.08e-04
[ep35 s14500] L=1.4314 L_pred=1.3955 L_idm=0.0007 ac_lift=+0.2544 gate=11.421 lr=3.03e-04

Epoch 35  loss=1.4591  ac_lift=+0.2664  idm=0.0007
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  → Saved: checkpoints\graph_wm\graph_wm_pusht_best.pt
[ep36 s14550] L=1.4702 L_pred=1.4359 L_idm=0.0007 ac_lift=+0.2682 gate=11.425 lr=2.99e-04
[ep36 s14600] L=1.4323 L_pred=1.4007 L_idm=0.0006 ac_lift=+0.2651 gate=11.425 lr=2.94e-04
[ep36 s14650] L=1.4754 L_pred=1.4394 L_idm=0.0007 ac_lift=+0.2528 gate=11.425 lr=2.89e-04
[ep36 s14700] L=1.4764 L_pred=1.4412 L_idm=0.0007 ac_lift=+0.2500 gate=11.424 lr=2.84e-04
[ep36 s14750] L=1.4617 L_pred=1.4265 L_idm=0.0007 ac_lift=+0.2513 gate=11.426 lr=2.79e-04
[ep36 s14800] L=1.4754 L_pred=1.4423 L_idm=0.0007 ac_lift=+0.2714 gate=11.426 lr=2.74e-04
[ep36 s14850] L=1.4304 L_pred=1.4002 L_idm=0.0006 ac_lift=+0.2884 gate=11.424 lr=2.70e-04
[ep36 s14900] L=1.4891 L_pred=1.4544 L_idm=0.0007 ac_lift=+0.2767 gate=11.426 lr=2.65e-04

Epoch 36  loss=1.4599  ac_lift=+0.2647  idm=0.0007
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
[ep37 s14950] L=1.4102 L_pred=1.3789 L_idm=0.0006 ac_lift=+0.2835 gate=11.425 lr=2.60e-04
[ep37 s15000] L=1.4416 L_pred=1.4052 L_idm=0.0007 ac_lift=+0.2480 gate=11.428 lr=2.56e-04
[ep37 s15050] L=1.4496 L_pred=1.4153 L_idm=0.0007 ac_lift=+0.2585 gate=11.426 lr=2.51e-04
[ep37 s15100] L=1.4996 L_pred=1.4621 L_idm=0.0007 ac_lift=+0.2640 gate=11.428 lr=2.46e-04
[ep37 s15150] L=1.4132 L_pred=1.3818 L_idm=0.0006 ac_lift=+0.2717 gate=11.433 lr=2.42e-04
[ep37 s15200] L=1.5109 L_pred=1.4744 L_idm=0.0007 ac_lift=+0.2690 gate=11.432 lr=2.37e-04
[ep37 s15250] L=1.5060 L_pred=1.4722 L_idm=0.0007 ac_lift=+0.2542 gate=11.431 lr=2.33e-04
[ep37 s15300] L=1.4247 L_pred=1.3915 L_idm=0.0007 ac_lift=+0.2809 gate=11.436 lr=2.29e-04
[ep37 s15350] L=1.4537 L_pred=1.4176 L_idm=0.0007 ac_lift=+0.2612 gate=11.434 lr=2.24e-04

Epoch 37  loss=1.4586  ac_lift=+0.2647  idm=0.0007
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  → Saved: checkpoints\graph_wm\graph_wm_pusht_best.pt
[ep38 s15400] L=1.4760 L_pred=1.4391 L_idm=0.0007 ac_lift=+0.2593 gate=11.432 lr=2.20e-04
[ep38 s15450] L=1.4159 L_pred=1.3810 L_idm=0.0007 ac_lift=+0.2708 gate=11.433 lr=2.15e-04
[ep38 s15500] L=1.5048 L_pred=1.4691 L_idm=0.0007 ac_lift=+0.2585 gate=11.435 lr=2.11e-04
[ep38 s15550] L=1.4691 L_pred=1.4335 L_idm=0.0007 ac_lift=+0.2729 gate=11.435 lr=2.07e-04
[ep38 s15600] L=1.4209 L_pred=1.3873 L_idm=0.0007 ac_lift=+0.2704 gate=11.435 lr=2.03e-04
[ep38 s15650] L=1.4711 L_pred=1.4357 L_idm=0.0007 ac_lift=+0.2838 gate=11.435 lr=1.99e-04
[ep38 s15700] L=1.4153 L_pred=1.3804 L_idm=0.0007 ac_lift=+0.2671 gate=11.436 lr=1.94e-04
[ep38 s15750] L=1.4738 L_pred=1.4410 L_idm=0.0007 ac_lift=+0.2743 gate=11.438 lr=1.90e-04

Epoch 38  loss=1.4586  ac_lift=+0.2696  idm=0.0007
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  → Saved: checkpoints\graph_wm\graph_wm_pusht_best.pt
[ep39 s15800] L=1.4578 L_pred=1.4265 L_idm=0.0006 ac_lift=+0.2637 gate=11.438 lr=1.86e-04
[ep39 s15850] L=1.4184 L_pred=1.3861 L_idm=0.0006 ac_lift=+0.2673 gate=11.437 lr=1.82e-04
[ep39 s15900] L=1.4516 L_pred=1.4161 L_idm=0.0007 ac_lift=+0.2600 gate=11.438 lr=1.78e-04
[ep39 s15950] L=1.4550 L_pred=1.4182 L_idm=0.0007 ac_lift=+0.2835 gate=11.439 lr=1.74e-04
[ep39 s16000] L=1.4383 L_pred=1.4055 L_idm=0.0007 ac_lift=+0.2680 gate=11.439 lr=1.70e-04
[ep39 s16050] L=1.4660 L_pred=1.4322 L_idm=0.0007 ac_lift=+0.2662 gate=11.437 lr=1.66e-04
[ep39 s16100] L=1.4541 L_pred=1.4188 L_idm=0.0007 ac_lift=+0.2681 gate=11.440 lr=1.63e-04
[ep39 s16150] L=1.4616 L_pred=1.4267 L_idm=0.0007 ac_lift=+0.2566 gate=11.439 lr=1.59e-04

Epoch 39  loss=1.4590  ac_lift=+0.2689  idm=0.0007
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  ✓ Block edge dominates goal edge (contact-aware routing)
[ep40 s16200] L=1.5363 L_pred=1.5004 L_idm=0.0007 ac_lift=+0.2667 gate=11.442 lr=1.55e-04
[ep40 s16250] L=1.4468 L_pred=1.4139 L_idm=0.0007 ac_lift=+0.2673 gate=11.442 lr=1.51e-04
[ep40 s16300] L=1.4295 L_pred=1.4006 L_idm=0.0006 ac_lift=+0.2730 gate=11.441 lr=1.48e-04
[ep40 s16350] L=1.4125 L_pred=1.3818 L_idm=0.0006 ac_lift=+0.2559 gate=11.440 lr=1.44e-04
[ep40 s16400] L=1.4549 L_pred=1.4238 L_idm=0.0006 ac_lift=+0.2637 gate=11.442 lr=1.40e-04
[ep40 s16450] L=1.4332 L_pred=1.4005 L_idm=0.0007 ac_lift=+0.2731 gate=11.443 lr=1.37e-04
[ep40 s16500] L=1.4266 L_pred=1.3971 L_idm=0.0006 ac_lift=+0.2647 gate=11.443 lr=1.33e-04
[ep40 s16550] L=1.4588 L_pred=1.4225 L_idm=0.0007 ac_lift=+0.2662 gate=11.443 lr=1.30e-04

Epoch 40  loss=1.4595  ac_lift=+0.2665  idm=0.0007
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
[ep41 s16600] L=1.4425 L_pred=1.4106 L_idm=0.0006 ac_lift=+0.2672 gate=11.443 lr=1.26e-04
[ep41 s16650] L=1.4484 L_pred=1.4158 L_idm=0.0007 ac_lift=+0.2639 gate=11.441 lr=1.23e-04
[ep41 s16700] L=1.4955 L_pred=1.4620 L_idm=0.0007 ac_lift=+0.2606 gate=11.441 lr=1.20e-04
[ep41 s16750] L=1.4618 L_pred=1.4312 L_idm=0.0006 ac_lift=+0.2582 gate=11.442 lr=1.16e-04
[ep41 s16800] L=1.4825 L_pred=1.4496 L_idm=0.0007 ac_lift=+0.2714 gate=11.442 lr=1.13e-04
[ep41 s16850] L=1.4253 L_pred=1.3932 L_idm=0.0006 ac_lift=+0.2679 gate=11.443 lr=1.10e-04
[ep41 s16900] L=1.4570 L_pred=1.4229 L_idm=0.0007 ac_lift=+0.2709 gate=11.445 lr=1.07e-04
[ep41 s16950] L=1.4348 L_pred=1.4038 L_idm=0.0006 ac_lift=+0.2630 gate=11.447 lr=1.04e-04

Epoch 41  loss=1.4576  ac_lift=+0.2679  idm=0.0007
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  → Saved: checkpoints\graph_wm\graph_wm_pusht_best.pt
[ep42 s17000] L=1.4204 L_pred=1.3896 L_idm=0.0006 ac_lift=+0.2546 gate=11.445 lr=1.01e-04
[ep42 s17050] L=1.4850 L_pred=1.4500 L_idm=0.0007 ac_lift=+0.2777 gate=11.447 lr=9.75e-05
[ep42 s17100] L=1.4191 L_pred=1.3848 L_idm=0.0007 ac_lift=+0.2637 gate=11.447 lr=9.45e-05
[ep42 s17150] L=1.4769 L_pred=1.4449 L_idm=0.0006 ac_lift=+0.2671 gate=11.446 lr=9.15e-05
[ep42 s17200] L=1.4634 L_pred=1.4279 L_idm=0.0007 ac_lift=+0.2731 gate=11.447 lr=8.86e-05
[ep42 s17250] L=1.4479 L_pred=1.4101 L_idm=0.0008 ac_lift=+0.2678 gate=11.447 lr=8.57e-05
[ep42 s17300] L=1.4262 L_pred=1.3915 L_idm=0.0007 ac_lift=+0.2738 gate=11.446 lr=8.29e-05
[ep42 s17350] L=1.5124 L_pred=1.4757 L_idm=0.0007 ac_lift=+0.2631 gate=11.446 lr=8.01e-05

Epoch 42  loss=1.4587  ac_lift=+0.2688  idm=0.0007
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
[ep43 s17400] L=1.4514 L_pred=1.4225 L_idm=0.0006 ac_lift=+0.2671 gate=11.446 lr=7.74e-05
[ep43 s17450] L=1.4643 L_pred=1.4311 L_idm=0.0007 ac_lift=+0.2628 gate=11.446 lr=7.47e-05
[ep43 s17500] L=1.4600 L_pred=1.4275 L_idm=0.0006 ac_lift=+0.2750 gate=11.446 lr=7.20e-05
[ep43 s17550] L=1.5154 L_pred=1.4812 L_idm=0.0007 ac_lift=+0.2695 gate=11.446 lr=6.94e-05
[ep43 s17600] L=1.4612 L_pred=1.4276 L_idm=0.0007 ac_lift=+0.2719 gate=11.447 lr=6.69e-05
[ep43 s17650] L=1.4273 L_pred=1.3950 L_idm=0.0006 ac_lift=+0.2506 gate=11.446 lr=6.44e-05
[ep43 s17700] L=1.4548 L_pred=1.4212 L_idm=0.0007 ac_lift=+0.2777 gate=11.447 lr=6.19e-05
[ep43 s17750] L=1.4846 L_pred=1.4516 L_idm=0.0007 ac_lift=+0.2621 gate=11.447 lr=5.95e-05

Epoch 43  loss=1.4584  ac_lift=+0.2679  idm=0.0007
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
[ep44 s17800] L=1.4465 L_pred=1.4124 L_idm=0.0007 ac_lift=+0.2665 gate=11.448 lr=5.71e-05
[ep44 s17850] L=1.4607 L_pred=1.4311 L_idm=0.0006 ac_lift=+0.2672 gate=11.447 lr=5.48e-05
[ep44 s17900] L=1.4219 L_pred=1.3920 L_idm=0.0006 ac_lift=+0.2605 gate=11.447 lr=5.25e-05
[ep44 s17950] L=1.4317 L_pred=1.3969 L_idm=0.0007 ac_lift=+0.2773 gate=11.447 lr=5.03e-05
[ep44 s18000] L=1.4471 L_pred=1.4128 L_idm=0.0007 ac_lift=+0.2797 gate=11.448 lr=4.81e-05
[ep44 s18050] L=1.4446 L_pred=1.4133 L_idm=0.0006 ac_lift=+0.2755 gate=11.447 lr=4.59e-05
[ep44 s18100] L=1.4081 L_pred=1.3738 L_idm=0.0007 ac_lift=+0.2575 gate=11.447 lr=4.39e-05
[ep44 s18150] L=1.4113 L_pred=1.3790 L_idm=0.0006 ac_lift=+0.2778 gate=11.447 lr=4.18e-05

Epoch 44  loss=1.4589  ac_lift=+0.2698  idm=0.0007
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
[ep45 s18200] L=1.5014 L_pred=1.4702 L_idm=0.0006 ac_lift=+0.2651 gate=11.448 lr=3.98e-05
[ep45 s18250] L=1.4475 L_pred=1.4177 L_idm=0.0006 ac_lift=+0.2653 gate=11.447 lr=3.79e-05
[ep45 s18300] L=1.4397 L_pred=1.4079 L_idm=0.0006 ac_lift=+0.2613 gate=11.447 lr=3.60e-05
[ep45 s18350] L=1.4608 L_pred=1.4262 L_idm=0.0007 ac_lift=+0.2670 gate=11.447 lr=3.41e-05
[ep45 s18400] L=1.4206 L_pred=1.3883 L_idm=0.0006 ac_lift=+0.2675 gate=11.447 lr=3.23e-05
[ep45 s18450] L=1.4600 L_pred=1.4301 L_idm=0.0006 ac_lift=+0.2870 gate=11.447 lr=3.05e-05
[ep45 s18500] L=1.4400 L_pred=1.4020 L_idm=0.0008 ac_lift=+0.2754 gate=11.447 lr=2.88e-05
[ep45 s18550] L=1.4983 L_pred=1.4671 L_idm=0.0006 ac_lift=+0.2674 gate=11.447 lr=2.72e-05

Epoch 45  loss=1.4589  ac_lift=+0.2683  idm=0.0007
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
[ep46 s18600] L=1.4575 L_pred=1.4260 L_idm=0.0006 ac_lift=+0.2681 gate=11.447 lr=2.56e-05
[ep46 s18650] L=1.4856 L_pred=1.4469 L_idm=0.0008 ac_lift=+0.2744 gate=11.447 lr=2.40e-05
[ep46 s18700] L=1.4403 L_pred=1.4083 L_idm=0.0006 ac_lift=+0.2752 gate=11.447 lr=2.25e-05
[ep46 s18750] L=1.4632 L_pred=1.4309 L_idm=0.0006 ac_lift=+0.2693 gate=11.447 lr=2.10e-05
[ep46 s18800] L=1.4375 L_pred=1.4053 L_idm=0.0006 ac_lift=+0.2631 gate=11.447 lr=1.96e-05
[ep46 s18850] L=1.4278 L_pred=1.3941 L_idm=0.0007 ac_lift=+0.2591 gate=11.447 lr=1.82e-05
[ep46 s18900] L=1.4206 L_pred=1.3900 L_idm=0.0006 ac_lift=+0.2698 gate=11.447 lr=1.69e-05
[ep46 s18950] L=1.4385 L_pred=1.4012 L_idm=0.0007 ac_lift=+0.2805 gate=11.448 lr=1.56e-05

Epoch 46  loss=1.4578  ac_lift=+0.2700  idm=0.0007
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  ✓ Block edge dominates goal edge (contact-aware routing)
[ep47 s19000] L=1.4584 L_pred=1.4254 L_idm=0.0007 ac_lift=+0.2790 gate=11.448 lr=1.44e-05
[ep47 s19050] L=1.5132 L_pred=1.4814 L_idm=0.0006 ac_lift=+0.2711 gate=11.447 lr=1.32e-05
[ep47 s19100] L=1.4896 L_pred=1.4553 L_idm=0.0007 ac_lift=+0.2732 gate=11.447 lr=1.21e-05
[ep47 s19150] L=1.4799 L_pred=1.4476 L_idm=0.0006 ac_lift=+0.2686 gate=11.447 lr=1.10e-05
[ep47 s19200] L=1.4718 L_pred=1.4397 L_idm=0.0006 ac_lift=+0.2765 gate=11.447 lr=1.00e-05
[ep47 s19250] L=1.4108 L_pred=1.3768 L_idm=0.0007 ac_lift=+0.2661 gate=11.447 lr=9.04e-06
[ep47 s19300] L=1.4168 L_pred=1.3837 L_idm=0.0007 ac_lift=+0.2738 gate=11.447 lr=8.11e-06
[ep47 s19350] L=1.4299 L_pred=1.3930 L_idm=0.0007 ac_lift=+0.2638 gate=11.447 lr=7.24e-06

Epoch 47  loss=1.4593  ac_lift=+0.2692  idm=0.0007
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
[ep48 s19400] L=1.4542 L_pred=1.4195 L_idm=0.0007 ac_lift=+0.2740 gate=11.447 lr=6.41e-06
[ep48 s19450] L=1.4455 L_pred=1.4146 L_idm=0.0006 ac_lift=+0.2769 gate=11.447 lr=5.64e-06
[ep48 s19500] L=1.4284 L_pred=1.3966 L_idm=0.0006 ac_lift=+0.2784 gate=11.447 lr=4.91e-06
[ep48 s19550] L=1.4609 L_pred=1.4226 L_idm=0.0008 ac_lift=+0.2748 gate=11.447 lr=4.24e-06
[ep48 s19600] L=1.4958 L_pred=1.4618 L_idm=0.0007 ac_lift=+0.2609 gate=11.447 lr=3.61e-06
[ep48 s19650] L=1.5385 L_pred=1.5082 L_idm=0.0006 ac_lift=+0.2553 gate=11.447 lr=3.03e-06
[ep48 s19700] L=1.4340 L_pred=1.3989 L_idm=0.0007 ac_lift=+0.2598 gate=11.447 lr=2.51e-06
[ep48 s19750] L=1.4594 L_pred=1.4262 L_idm=0.0007 ac_lift=+0.2615 gate=11.447 lr=2.03e-06

Epoch 48  loss=1.4580  ac_lift=+0.2702  idm=0.0007
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
[ep49 s19800] L=1.4579 L_pred=1.4236 L_idm=0.0007 ac_lift=+0.2600 gate=11.447 lr=1.61e-06
[ep49 s19850] L=1.4681 L_pred=1.4327 L_idm=0.0007 ac_lift=+0.2693 gate=11.447 lr=1.23e-06
[ep49 s19900] L=1.4536 L_pred=1.4236 L_idm=0.0006 ac_lift=+0.2696 gate=11.447 lr=9.04e-07
[ep49 s19950] L=1.4493 L_pred=1.4164 L_idm=0.0007 ac_lift=+0.2834 gate=11.447 lr=6.29e-07
[ep49 s20000] L=1.4624 L_pred=1.4321 L_idm=0.0006 ac_lift=+0.2676 gate=11.447 lr=4.04e-07
[ep49 s20050] L=1.4345 L_pred=1.4008 L_idm=0.0007 ac_lift=+0.2596 gate=11.447 lr=2.29e-07
[ep49 s20100] L=1.4569 L_pred=1.4228 L_idm=0.0007 ac_lift=+0.2654 gate=11.447 lr=1.05e-07
[ep49 s20150] L=1.4990 L_pred=1.4666 L_idm=0.0006 ac_lift=+0.2749 gate=11.447 lr=3.01e-08
[ep49 s20200] L=1.4465 L_pred=1.4077 L_idm=0.0008 ac_lift=+0.2753 gate=11.447 lr=6.01e-09

Epoch 49  loss=1.4596  ac_lift=+0.2697  idm=0.0007
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  ep01 ✗  dist=0.493
  ep02 ✗  dist=0.853
  ep03 ✗  dist=0.551
  ep04 ✗  dist=0.663
  ep05 ✗  dist=0.532
  ep06 ✗  dist=0.643
  ep07 ✗  dist=0.425
  ep08 ✗  dist=0.779
  ep09 ✗  dist=0.491
  ep10 ✗  dist=0.618
  ep11 ✗  dist=0.381
  ep12 ✗  dist=0.635
  ep13 ✗  dist=0.395
  ep14 ✗  dist=0.451
  ep15 ✗  dist=0.383
  ep16 ✗  dist=0.650
  ep17 ✗  dist=0.592
  ep18 ✗  dist=0.454
  ep19 ✗  dist=0.467
  ep20 ✗  dist=0.398
  ep21 ✗  dist=0.259
  ep22 ✗  dist=0.589
  ep23 ✗  dist=0.420
  ep24 ✗  dist=0.597
  ep25 ✗  dist=0.617
  ep26 ✗  dist=0.265
  ep27 ✗  dist=0.470
  ep28 ✗  dist=0.360
  ep29 ✗  dist=0.385
  ep30 ✗  dist=0.204
  ep31 ✗  dist=0.481
  ep32 ✗  dist=0.515
  ep33 ✗  dist=0.432
  ep34 ✗  dist=0.492
  
  
  need to fix agent=0.333
  
  r=1.23e-06
[ep49 s19900] L=1.5604 L_pred=1.5250 L_idm=0.0007 ac_lift=+0.2424 gate=11.176 lr=9.04e-07
[ep49 s19950] L=1.5699 L_pred=1.5362 L_idm=0.0007 ac_lift=+0.2360 gate=11.176 lr=6.29e-07
[ep49 s20000] L=1.5701 L_pred=1.5382 L_idm=0.0006 ac_lift=+0.2248 gate=11.176 lr=4.04e-07
[ep49 s20050] L=1.5335 L_pred=1.4997 L_idm=0.0007 ac_lift=+0.2357 gate=11.176 lr=2.29e-07
[ep49 s20100] L=1.5449 L_pred=1.5137 L_idm=0.0006 ac_lift=+0.2454 gate=11.176 lr=1.05e-07
[ep49 s20150] L=1.5472 L_pred=1.5157 L_idm=0.0006 ac_lift=+0.2489 gate=11.176 lr=3.01e-08
[ep49 s20200] L=1.5425 L_pred=1.5105 L_idm=0.0006 ac_lift=+0.2320 gate=11.176 lr=6.01e-09

Epoch 49  loss=1.5562  ac_lift=+0.2388  idm=0.0007
  Edge attention: agent=0.333  block=0.333  goal=0.333
  ✓ Action conditioning load-bearing
  ep01 ✗  dist=0.440
  ep02 ✗  dist=0.554
  ep03 ✗  dist=0.532
  ep04 ✗  dist=0.427
  ep05 ✗  dist=0.295
  ep06 ✗  dist=0.472
  ep07 ✗  dist=0.611
  ep08 ✗  dist=0.436
  ep09 ✗  dist=0.618
  ep10 ✗  dist=0.704
  ep11 ✗  dist=0.245
  ep12 ✗  dist=0.583
  ep13 ✗  dist=0.604
  ep14 ✗  dist=0.502
  ep15 ✗  dist=0.384
  ep16 ✗  dist=0.355
  ep17 ✗  dist=0.687
  ep18 ✗  dist=0.472
  ep19 ✗  dist=0.517
  ep20 ✗  dist=0.351
  ep21 ✓  dist=0.078
  ep22 ✗  dist=0.140
  ep23 ✗  dist=0.356
  ep24 ✗  dist=0.253
  ep25 ✗  dist=0.680
  
  
    ep43 ✗  dist=0.606
  ep44 ✗  dist=0.466
  ep45 ✗  dist=0.247
  ep46 ✗  dist=0.444
  ep47 ✗  dist=0.654
  ep48 ✗  dist=0.308
  ep49 ✗  dist=0.273
  ep50 ✗  dist=0.578

══ Graph WM SR: 0.0% (0/50) ══
(ryzen-ai-1.7.0) PS C:\Users\MeteorAI\Desktop\CORTEX>
  # Run flat MPC eval to get the comparison number
python eval_pusht_sr.py \
    --ckpt checkpoints/action_wm/action_wm_pusht_full_best.pt \
    --n-episodes 50 --no-video --synthetic
    
    
    (ryzen-ai-1.7.0) PS C:\Users\MeteorAI\Desktop\CORTEX> Copy-Item "$env:USERPROFILE\Downloads\eval_pusht_sr.py" "C:\Users\MeteorAI\Desktop\CORTEX\"
Copy-Item : Cannot find path 'C:\Users\MeteorAI\Downloads\eval_pusht_sr.py'
because it does not exist.
At line:1 char:1
+ Copy-Item "$env:USERPROFILE\Downloads\eval_pusht_sr.py" "C:\Users\Met ...
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : ObjectNotFound: (C:\Users\Meteor...val_pusht_sr.py:S
   tring) [Copy-Item], ItemNotFoundException
    + FullyQualifiedErrorId : PathNotFound,Microsoft.PowerShell.Commands.CopyItemC
   ommand

(ryzen-ai-1.7.0) PS C:\Users\MeteorAI\Desktop\CORTEX>
(ryzen-ai-1.7.0) PS C:\Users\MeteorAI\Desktop\CORTEX> python eval_pusht_sr.py `
>>     --ckpt checkpoints/action_wm/action_wm_pusht_full_best.pt `
>>     --n-episodes 50 --no-video --synthetic
Loaded: checkpoints/action_wm/action_wm_pusht_full_best.pt
  obs_dim=5 action_dim=2 k_steps=4
  epoch=45 loss=0.6120
  ac_lift=+0.6536

── WM Planner Synthetic (K=512, H=8) ──
  ep01 ✗  steps=300  dist=0.468  DA=0.43  🧊 COLD
  ep02 ✗  steps=300  dist=0.308  DA=0.39  🧊 COLD
  ep03 ✗  steps=300  dist=0.506  DA=0.34  ❄  COOL
  ep04 ✗  steps=300  dist=0.282  DA=0.46  ❄  COOL
  ep05 ✗  steps=300  dist=0.405  DA=0.23  〜 NEUTRAL
  ep06 ✗  steps=300  dist=0.213  DA=0.17  🧊 COLD
  ep07 ✗  steps=300  dist=0.344  DA=0.46  ❄  COOL
  ep08 ✗  steps=300  dist=0.454  DA=0.44  ❄  COOL
  ep09 ✗  steps=300  dist=0.394  DA=0.27  🧊 COLD
  ep10 ✗  steps=300  dist=0.256  DA=0.28  〜 NEUTRAL
  ep11 ✗  steps=300  dist=0.235  DA=0.35  🔥 HOT
  ep12 ✗  steps=300  dist=0.447  DA=0.41  🧊 COLD
  ep13 ✗  steps=300  dist=0.389  DA=0.30  ❄  COOL
  ep14 ✗  steps=300  dist=0.269  DA=0.30  ♨  WARM
  ep15 ✗  steps=300  dist=0.258  DA=0.45  ❄  COOL
  ep16 ✗  steps=300  dist=0.207  DA=0.27  🧊 COLD
  ep17 ✗  steps=300  dist=0.299  DA=0.44  🧊 COLD
  
  