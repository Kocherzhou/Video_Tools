import json, glob, os

session_dir = r'C:\Users\tanga\.openclaw\agents\main\sessions'
files = sorted(glob.glob(session_dir + '/*.jsonl'), key=os.path.getmtime)

for fpath in files:
    size = os.path.getsize(fpath)
    if size < 5000:
        continue
    try:
        with open(fpath, encoding='utf-8', errors='replace') as f:
            content = f.read()
        markers = ['SUBTITLE_PROMPT', 'generate_subtitles', 'slot_seconds',
                   '4.5字/秒', '4.5 chars', 'semantic unit', '语义单元']
        for m in markers:
            if m in content:
                idx = content.find(m)
                fname = os.path.basename(fpath)
                snippet = content[max(0,idx-100):idx+500]
                print(f'=== {fname} | marker: {m} ===')
                print(snippet[:600])
                print()
                break
    except Exception as e:
        pass
