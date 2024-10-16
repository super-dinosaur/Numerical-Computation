import json
import numpy as np

prompts = {
    'a':  0,
    'b':  1,
    'epsilon': 0.5*10**-5
}

with open('params.json', 'w') as f:
    json.dump(prompts, f, indent=4)