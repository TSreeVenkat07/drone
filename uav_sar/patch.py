with open('airsim_wrapper.py', 'r') as f:
    c = f.read()

import re

# Change loops to only run for 1 agent
c = re.sub(r'for i in range\(1, 4\):', r'for i in range(1, 1):', c)
c = re.sub(r'for i in range\(4\):', r'for i in range(1):', c)

# Replace any explicit drone_0 string
c = c.replace('"drone_0"', '"SimpleFlight"')
c = c.replace("'drone_0'", "'SimpleFlight'")

# Fix dynamic f-strings
c = c.replace('f"drone_{i}"', '("SimpleFlight" if i==0 else f"drone_{i}")')

with open('airsim_wrapper.py', 'w') as f:
    f.write(c)
