import re, os
with open('../writeup.txt', 'r', encoding='utf-8') as f:
    s = f.read()

# Look for any remaining "Pan" references
print('Remaining occurrences of "Pan ":')
for m in re.finditer(r' Pan ', s):
    start = max(0, m.start()-50); end = min(len(s), m.end()+50)
    print('  ', repr(s[start:end]))

# Find figure refs and includegraphics
figs = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", s)
print()
print('Figures referenced (', len(figs), '):')
for f in figs:
    print('  ', f)

present = os.listdir('../figures')
missing = []
for f in figs:
    name = os.path.basename(f)
    if name not in present:
        missing.append(f)
print()
print('Missing figures:', missing or 'NONE')

# Tables
tabs = re.findall(r"\\begin\{table\}", s)
print('Tables:', len(tabs))

# Sections
secs = re.findall(r"\\section\{([^}]+)\}", s)
print('Sections:')
for sc in secs:
    print('  -', sc)

print()
print('Total chars:', len(s))
print('Lines:', s.count(chr(10)))
