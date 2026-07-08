content = open('C:/Users/tanga/video-subtitle/app.py', encoding='utf-8', errors='replace').read()
old = 'f"ass={ass_path.resolve()}"'
new = 'f"ass={escape_ass_path(ass_path)}"'
count = content.count(old)
print(f"Found {count} occurrences of old string")
fixed = content.replace(old, new)
open('C:/Users/tanga/video-subtitle/app.py', 'w', encoding='utf-8').write(fixed)
print(f"Done. {fixed.count('escape_ass_path')} escape_ass_path found in file")
