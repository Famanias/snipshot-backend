import csv
import statistics

# Read and parse CSV
with open('e:/Downloads/SnipShot_ A Free Screen-Snipping Translation Tool.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f'Total Responses: {len(rows)}')
print()

# Analyze user demographics
user_types = {}
devices = {}
content_types = {}

for row in rows:
    # User groups
    groups = row['Which group best describes you? (Choose all that apply)']
    for g in groups.split(';'):
        g = g.strip()
        if g:
            user_types[g] = user_types.get(g, 0) + 1
    
    # Devices
    devs = row['Which device do you primarily use when reading online? (Choose all that apply)']
    for d in devs.split(';'):
        d = d.strip()
        if d:
            devices[d] = devices.get(d, 0) + 1

print('=== USER DEMOGRAPHICS ===')
print('User Types:')
for k, v in sorted(user_types.items(), key=lambda x: -x[1]):
    print(f'  {k}: {v} ({v/len(rows)*100:.1f}%)')

print()
print('Devices:')
for k, v in sorted(devices.items(), key=lambda x: -x[1]):
    print(f'  {k}: {v} ({v/len(rows)*100:.1f}%)')

# Numeric analysis
def get_numeric(rows, col):
    vals = []
    for r in rows:
        try:
            v = int(r[col])
            vals.append(v)
        except:
            pass
    return vals

print()
print('=== CURRENT TOOL SATISFACTION ===')
satisfaction = get_numeric(rows, 'How satisfied are you with the current tools for translating images containing texts/characters?')
print(f'Satisfaction with current tools: Mean={statistics.mean(satisfaction):.2f}, SD={statistics.stdev(satisfaction):.2f}, n={len(satisfaction)}')

difficulty = get_numeric(rows, 'How often do you face difficulties accessing translation tools for images containing texts/characters?')
print(f'Frequency of difficulties: Mean={statistics.mean(difficulty):.2f}, SD={statistics.stdev(difficulty):.2f}, n={len(difficulty)}')

ease = get_numeric(rows, "How easy is it to use current tools to translate texts from your device's screen?")
print(f'Ease of current tools: Mean={statistics.mean(ease):.2f}, SD={statistics.stdev(ease):.2f}, n={len(ease)}')

print()
print('=== SNIPSHOT EVALUATION ===')
concept = get_numeric(rows, 'Now that you\'ve seen the wireframe, how well do you understand the concept of Snipshot?')
print(f'Concept Understanding: Mean={statistics.mean(concept):.2f}, SD={statistics.stdev(concept):.2f}, n={len(concept)}')

design = get_numeric(rows, 'How would you rate the wireframe design of Snipshot?')
print(f'Design Rating: Mean={statistics.mean(design):.2f}, SD={statistics.stdev(design):.2f}, n={len(design)}')

usefulness = get_numeric(rows, 'How useful would an app be that lets you select an area from your screen (like snipping tool) containing texts or characters and translates it to your desired language instantly?')
print(f'Perceived Usefulness: Mean={statistics.mean(usefulness):.2f}, SD={statistics.stdev(usefulness):.2f}, n={len(usefulness)}')

improvement = get_numeric(rows, 'How much would SnipShot improve your ability to understand images containing different language on your desktop/mobile screen?')
print(f'Expected Improvement: Mean={statistics.mean(improvement):.2f}, SD={statistics.stdev(improvement):.2f}, n={len(improvement)}')

# Usage frequency
print()
print('=== BEHAVIORAL INTENTION ===')
usage_freq = {}
for row in rows:
    f = row['How often would you use an app like SnipShot for translating images containing texts/characters?']
    usage_freq[f] = usage_freq.get(f, 0) + 1
for k, v in sorted(usage_freq.items(), key=lambda x: -x[1]):
    print(f'  {k}: {v} ({v/len(rows)*100:.1f}%)')

# Challenges
print()
print('=== CHALLENGES WITH CURRENT TOOLS ===')
challenges = {}
for row in rows:
    challs = row['What challenges do you encounter with current translation tools for screen text? (Check all that apply)']
    for c in challs.split(';'):
        c = c.strip()
        if c:
            challenges[c] = challenges.get(c, 0) + 1
for k, v in sorted(challenges.items(), key=lambda x: -x[1]):
    print(f'  {v} ({v/len(rows)*100:.1f}%): {k}')

# Best features
print()
print('=== BEST FEATURES OF SNIPSHOT ===')
features = {}
# Find the column with "best thing" in its name
best_col = [k for k in rows[0].keys() if 'best thing' in k.lower()][0]
for row in rows:
    feats = row[best_col]
    for f in feats.split(';'):
        f = f.strip()
        if f:
            features[f] = features.get(f, 0) + 1
for k, v in sorted(features.items(), key=lambda x: -x[1]):
    print(f'  {v} ({v/len(rows)*100:.1f}%): {k}')

# Current tools used
print()
print('=== CURRENT TOOLS USED ===')
tools = {}
for row in rows:
    t_list = row['Do you currently use any tools to translate images containing texts/characters to your desired language? If yes, which ones?']
    for t in t_list.split(';'):
        t = t.strip()
        if t:
            tools[t] = tools.get(t, 0) + 1
for k, v in sorted(tools.items(), key=lambda x: -x[1]):
    print(f'  {v} ({v/len(rows)*100:.1f}%): {k}')

# Speed perception
print()
print('=== SPEED PERCEPTION ===')
speed = {}
for row in rows:
    s = row['Do you think translating text from images takes too long with current tools?']
    speed[s] = speed.get(s, 0) + 1
for k, v in sorted(speed.items(), key=lambda x: -x[1]):
    print(f'  {v} ({v/len(rows)*100:.1f}%): {k}')

# Digital reader content
print()
print('=== DIGITAL READER CONTENT ===')
for row in rows:
    content = row['If you checked \'Digital Reader\' in the previous question, which type(s) of content do you primarily read? (Choose all that apply)']
    if content:
        for c in content.split(';'):
            c = c.strip()
            if c:
                content_types[c] = content_types.get(c, 0) + 1
for k, v in sorted(content_types.items(), key=lambda x: -x[1]):
    print(f'  {v}: {k}')

# Distribution analysis for Likert scales
print()
print('=== LIKERT SCALE DISTRIBUTIONS ===')

def analyze_likert(rows, col, name):
    vals = get_numeric(rows, col)
    if not vals:
        return
    dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for v in vals:
        if v in dist:
            dist[v] += 1
    print(f'\n{name}:')
    print(f'  Mean: {statistics.mean(vals):.2f}, SD: {statistics.stdev(vals):.2f}, n={len(vals)}')
    print(f'  Distribution: 1={dist[1]}, 2={dist[2]}, 3={dist[3]}, 4={dist[4]}, 5={dist[5]}')
    print(f'  Agree+Strongly Agree (4-5): {dist[4]+dist[5]} ({(dist[4]+dist[5])/len(vals)*100:.1f}%)')

analyze_likert(rows, 'How satisfied are you with the current tools for translating images containing texts/characters?', 'Current Tool Satisfaction')
analyze_likert(rows, 'How often do you face difficulties accessing translation tools for images containing texts/characters?', 'Difficulty Frequency')
analyze_likert(rows, "How easy is it to use current tools to translate texts from your device's screen?", 'Current Tool Ease')
analyze_likert(rows, 'Now that you\'ve seen the wireframe, how well do you understand the concept of Snipshot?', 'Concept Understanding')
analyze_likert(rows, 'How would you rate the wireframe design of Snipshot?', 'Design Rating')
analyze_likert(rows, 'How useful would an app be that lets you select an area from your screen (like snipping tool) containing texts or characters and translates it to your desired language instantly?', 'Perceived Usefulness')
analyze_likert(rows, 'How much would SnipShot improve your ability to understand images containing different language on your desktop/mobile screen?', 'Expected Improvement')
