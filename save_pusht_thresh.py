import json
# Save just the pusht threshold to its own file
json.dump({"pusht": 0.42942}, open('./benchmark_thresholds_pusht.json', 'w'), indent=2)
print('Saved benchmark_thresholds_pusht.json')
