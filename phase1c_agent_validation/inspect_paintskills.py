from huggingface_hub import hf_hub_download
import json

scenes_path = hf_hub_download(
    'j-min/PaintSkills',
    'spatial/scenes/spatial_val.json',
    repo_type='dataset'
)
with open(scenes_path) as f:
    scenes = json.load(f)

print(f'Type: {type(scenes)}')
if isinstance(scenes, list):
    print(f'Total scenes: {len(scenes)}')
    print('First 3 scenes:')
    for s in scenes[:3]:
        print(s)
elif isinstance(scenes, dict):
    keys = list(scenes.keys())
    print(f'Top-level keys: {keys[:5]}')
    first = scenes[keys[0]]
    print(f'First entry type: {type(first)}')
    if isinstance(first, list):
        print(f'First entry length: {len(first)}')
        print('First 3 items:')
        for item in first[:3]:
            print(item)
    else:
        print(f'First entry: {first}')
