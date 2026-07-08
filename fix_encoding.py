content = open('C:/Users/tanga/video-subtitle/app.py', encoding='utf-8', errors='replace').read()

# Fix garbled Chinese lines
fixes = [
    ('f"瀹屾垚锛佸叡鐢熸垚 {len(subtitles)} 鏉″瓧骞?', 'f"完成！共生成 {len(subtitles)} 条字幕"'),
    ("f\"瀛楀箷鐢熸垚瀹屾垚锛屽叡 {len(subtitles)} 鏉?, \"success\"", 'f"字幕生成完成，共 {len(subtitles)} 条", "success"'),
]

for old, new in fixes:
    if old in content:
        content = content.replace(old, new)
        print(f"Fixed: {old[:30]}...")
    else:
        print(f"Not found: {old[:30]}...")

# Also scan for any remaining garbled sequences
import re
garbled = re.findall(r'[\u9500-\u9fff]{3,}', content)
if garbled:
    print(f"WARNING: {len(garbled)} remaining garbled sequences")
    for g in garbled[:5]:
        print(f"  {g}")
else:
    print("No remaining garbled text detected")

open('C:/Users/tanga/video-subtitle/app.py', 'w', encoding='utf-8').write(content)
print("Saved.")
