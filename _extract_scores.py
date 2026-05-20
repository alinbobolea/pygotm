import re
with open("/home/nick/projects/pygotm/validation/ows_papa-gotm.html", "rb") as f:
    raw = f.read()
text = raw.decode("utf-8", errors="ignore")
# Look for rows: <tr>...<td>BROKEN|DISCREPANT</td><td>VAR</td><td>REF</td><td>CALC</td><td>RAW</td><td>SCORE</td>
pattern = re.compile(
    r'(BROKEN|DISCREPANT)</td><td>([a-z_0-9]+)</td><td><code>([^<]+)</code></td><td><code>([^<]+)</code></td><td>([^<]+)</td><td>([^<]+)</td>'
)
seen = set()
rows = []
for m in pattern.finditer(text):
    status, var, ref, calc, raw_d, score = m.groups()
    key = (status, var)
    if key in seen:
        continue
    seen.add(key)
    rows.append((status, var, ref, calc, raw_d, score))

# Sort by score (descending)
def score_val(s):
    try:
        return float(s.strip())
    except Exception:
        return -1.0

rows.sort(key=lambda r: score_val(r[5]), reverse=True)
print(f"Total broken/discrepant rows: {len(rows)}\n")
for status, var, ref, calc, raw_d, score in rows:
    print(f"{status:11s} {var:12s} ref={ref:>22s} calc={calc:>22s} raw={raw_d:>11s} score={score}")
