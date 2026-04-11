import json, shutil
src = r'C:\Users\MeteorAI\Desktop\cortex-12v15\vaip_config.json'
dst = r'C:\Users\MeteorAI\Desktop\CORTEX\vaip_config.json'
shutil.copy(src, dst)
cfg = json.load(open(dst))
cfg['xclbin'] = r'C:\Users\MeteorAI\Downloads\NPU_RAI1.5_280_WHQL\npu_mcdm_stack_prod\AMD_AIE2P_4x4_Overlay_3.5.0.0-2354_ipu_2.xclbin'
cfg['compiler']['xcompiler_params']['target_alias'] = 'strix'
json.dump(cfg, open(dst, 'w'), indent=2)
print('ok - xclbin:', cfg['xclbin'])
