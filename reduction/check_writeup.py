import re
with open('../writeup.txt', 'r', encoding='utf-8') as f:
    s = f.read()

# bib
bibs = set(re.findall(r'\\bibitem\[.*?\]\{([^}]+)\}', s))
# citations
patterns = [
    r"\\citep\*?(?:\[[^\]]*\])?\{([^}]+)\}",
    r"\\citet\*?(?:\[[^\]]*\])?\{([^}]+)\}",
    r"\\cite\{([^}]+)\}",
    r"\\citealt\*?\{([^}]+)\}",
]
cites = set()
for p in patterns:
    for m in re.finditer(p, s):
        for k in m.group(1).split(','):
            cites.add(k.strip())

print('Bibitems found ({}):'.format(len(bibs)))
for b in sorted(bibs):
    print(' -', b)
print()
print('Citations found ({}):'.format(len(cites)))
for c in sorted(cites):
    print(' -', c)

print()
missing = cites - bibs
extra = bibs - cites
if missing:
    print('CITES NOT IN BIB:', missing)
if extra:
    print('BIB NOT CITED:', extra)

print('Total chars:', len(s))
print('Lines:', s.count(chr(10)))
