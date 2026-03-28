import re
import sys

path = r'c:\Users\KRISH\Desktop\Trading\src\dashboard\static\index.html'
try:
    old = open(path, 'r', encoding='utf8').read()
    new_style = open('new_style.css', 'r', encoding='utf8').read()
    res = re.sub(r'<style>.*?</style>', new_style, old, flags=re.DOTALL)
    open(path, 'w', encoding='utf8').write(res)
    print("Done, length:", len(res), "Matched:", "<style>" in new_style)
except Exception as e:
    print("Error:", e)
