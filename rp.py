import sys
from pathlib import Path

p = Path(r"c:\Users\KRISH\Desktop\Trading\src\dashboard\static\index.html")
lines = p.read_text("utf8").splitlines(True)
start = -1
end = -1
for i, l in enumerate(lines):
    if "<style>" in l:
        start = i
    if "</style>" in l:
        end = i
        break

new_css = Path("new_style.css").read_text("utf8")
if start != -1 and end != -1:
    # remove the <style> and </style> from new_css if it has them
    new_css = new_css.replace("<style>", "").replace("</style>", "")
    res = lines[:start+1] + [new_css + "\n"] + lines[end:]
    p.write_text("".join(res), "utf8")
    print("Success")
else:
    print("Tags not found")
