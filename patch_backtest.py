p = "src/eval/backtest.py"
t = open(p, encoding="utf-8").read()
old = '        drivers = e["driver"].tolist()'
new = ("        if len(e) < 5:\n"
       "            print(f'skip race {race_id}: {len(e)} entries')\n"
       "            continue\n"
       "        drivers = e[\"driver\"].tolist()")
if new in t:
    print("Already patched")
elif old not in t:
    print("MARKER NOT FOUND")
else:
    open(p, "w", encoding="utf-8").write(t.replace(old, new, 1))
    print("Patched OK")
