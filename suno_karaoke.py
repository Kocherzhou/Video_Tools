#!/usr/bin/env python3
# Suno 逐词时间戳 → 卡拉OK ASS。可选双语:英文逐词扫光在上 + 中文译文静态在下(按英文文本匹配)。
# 用法: suno_karaoke.py --video V --song-id ID --out O [--trans 双语.txt --font-size 40 ...]
import argparse, json, os, re, subprocess, urllib.request

def dims(v):
    o = subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries",
        "stream=width,height","-of","csv=p=0",v],capture_output=True,text=True).stdout.strip()
    w,h=o.split(",")[:2]; return int(w),int(h)
def cs(t):
    h=int(t//3600); m=int((t%3600)//60); s=t%60; return f"{h}:{m:02d}:{s:05.2f}"
def col(hx):
    hx=hx.lstrip("#"); r,g,b=hx[0:2],hx[2:4],hx[4:6]; return f"&H00{b}{g}{r}".upper()
def norm(s): return re.sub(r"[^a-z0-9]","",s.lower())
def cjk(s): return bool(re.search(r"[一-鿿]",s))

ap=argparse.ArgumentParser()
ap.add_argument("--video",required=True); ap.add_argument("--song-id",required=True)
ap.add_argument("--out",required=True); ap.add_argument("--suno",default="http://localhost:3000")
ap.add_argument("--aligned",default="")   # 本地对齐缓存 json(有则用、不依赖 suno-api)
ap.add_argument("--trans",default="")
ap.add_argument("--font",default="Noto Sans CJK SC"); ap.add_argument("--font-size",type=int,default=40)
ap.add_argument("--margin-v",type=int,default=54); ap.add_argument("--outline",type=float,default=2.5)
ap.add_argument("--primary",default="#33E0FF"); ap.add_argument("--secondary",default="#FFFFFF")
ap.add_argument("--lead-in",type=float,default=0.6); ap.add_argument("--linger",type=float,default=1.2)
ap.add_argument("--title",default=""); ap.add_argument("--title2",default="")
ap.add_argument("--credit",default=""); ap.add_argument("--credit2",default="")
a=ap.parse_args()

# 双语映射:英文行(归一)→中文
pairs={}
if a.trans and os.path.exists(a.trans):
    last=None
    for ln in open(a.trans,encoding="utf-8"):
        s=ln.strip()
        if not s or s.startswith("[") or (s and s[0] in "(（"): continue
        if cjk(s):
            if last: pairs[norm(last)]=s; last=None
        else: last=s
    print(f"双语映射 {len(pairs)} 行")

if a.aligned and os.path.exists(a.aligned):
    words=json.load(open(a.aligned,encoding="utf-8")); print(f"用对齐缓存 {a.aligned}")
else:
    _op=urllib.request.build_opener(urllib.request.ProxyHandler({}))   # 本地调用绕过 SOCKS5 代理
    words=json.loads(_op.open(f"{a.suno}/api/get_aligned_lyrics?song_id={a.song_id}",timeout=60).read())
lines=[]; cur=[]; in_paren=False
for w in words:
    raw=w.get("word",""); s=float(w.get("start_s",0)); e=float(w.get("end_s",0))
    for pi,part in enumerate(raw.split("\n")):
        if pi>0 and cur: lines.append(cur); cur=[]
        part=re.sub(r"\[[^\]]*\]","",part)
        buf=[]
        for ch in part:
            if ch in "(（": in_paren=True
            elif ch in ")）": in_paren=False
            elif not in_paren: buf.append(ch)
        txt="".join(buf).strip()
        if txt:
            for tok in txt.split(): cur.append((tok,s,e))
if cur: lines.append(cur)
lines=[ln for ln in lines if ln]
print(f"行数 {len(lines)}, 词数 {sum(len(l) for l in lines)}")

# 修 Suno 偶发对齐 bug:某行短到离谱(<0.25s/词)且下一行长到离谱(>1.2s/词)→ 两行词在合并段内均摊重排
def _ldur(l): return l[-1][2]-l[0][1]
fixed=0
for i in range(len(lines)-1):
    A,B=lines[i],lines[i+1]
    if _ldur(A)/max(1,len(A))<0.25 and _ldur(B)/max(1,len(B))>1.2:
        s0=A[0][1]; e1=B[-1][2]; tot=len(A)+len(B); step=(e1-s0)/tot; k=0
        for li in (A,B):
            for j in range(len(li)):
                li[j]=(li[j][0], s0+k*step, s0+(k+1)*step); k+=1
        fixed+=1
if fixed: print(f"修正 {fixed} 处 Suno 对齐异常(短行/长行均摊)")

W,H=dims(a.video)
ENG_MV = a.margin_v + (44 if pairs else 0)   # 有中文时英文行抬高,给下面中文留位
ZH_SIZE = int(a.font_size*0.72)
head=f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: K,{a.font},{a.font_size},{col(a.primary)},{col(a.secondary)},&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,{a.outline},1,2,40,40,{ENG_MV},1
Style: ZH,{a.font},{ZH_SIZE},{col(a.secondary)},{col(a.secondary)},&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,{max(1.5,a.outline-0.5)},1,2,40,40,{a.margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, Effect, Text
"""
rows=[(ln, ln[0][1], ln[-1][2]) for ln in lines]
disp_s=[max(ss-a.lead_in, rows[i-1][2] if i>0 else 0.0) for i,(_,ss,_) in enumerate(rows)]
disp_e=[]
for i,(_,ss,se) in enumerate(rows):
    nxt=disp_s[i+1] if i<len(rows)-1 else se+a.linger+1
    disp_e.append(min(se+a.linger, nxt-0.05))
ev=[]; matched=0
for i,(ln,ss,se) in enumerate(rows):
    ds,de=disp_s[i],disp_e[i]
    if de<=ds: de=ds+0.5
    starts=[t[1] for t in ln]
    kara=""; lead=int(round((ss-ds)*100))
    if lead>0: kara+=f"{{\\k{lead}}}"
    for j,(tok,ts,te) in enumerate(ln):
        nxt=starts[j+1] if j<len(starts)-1 else se
        d=max(1,int(round((nxt-ts)*100)))
        kara+=f"{{\\kf{d}}}{tok} "
    ev.append(f"Dialogue: 0,{cs(ds)},{cs(de)},K,,0,0,0,{kara.rstrip()}")
    zh=pairs.get(norm("".join(t[0] for t in ln)))
    if zh:
        ev.append(f"Dialogue: 0,{cs(ds)},{cs(de)},ZH,,0,0,0,{zh}"); matched+=1
if pairs: print(f"中文匹配 {matched}/{len(rows)} 行")
# 标题卡:前奏空镜上居中淡入淡出(歌名/声线/词曲署名),不挡歌词
if a.title:
    intro_end = rows[0][1] if rows else 8.0
    t0=1.0; t1=min(8.0, intro_end-1.2)
    if t1>t0+1.0:
        # 排版:歌名(大)/中文名(小) → 空两行 → 词曲 / 声线版(小)
        tx=f"{{\\an5\\pos({W//2},{int(H*0.42)})\\fad(700,700)\\bord2.6\\fs{int(a.font_size*1.5)}}}{a.title}"
        if a.title2: tx+=f"\\N{{\\fs{int(a.font_size*0.82)}}}{a.title2}"
        tx+="\\N\\N\\N"                                  # 空两行
        if a.credit:  tx+=f"{{\\fs{int(a.font_size*0.70)}}}{a.credit}"
        if a.credit2: tx+=f"\\N{{\\fs{int(a.font_size*0.56)}}}{a.credit2}"
        ev.insert(0, f"Dialogue: 0,{cs(t0)},{cs(t1)},K,,0,0,0,{tx}")
        print(f"标题卡 {t0:.0f}-{t1:.0f}s")
ass="/tmp/_suno_k.ass"; open(ass,"w",encoding="utf-8").write(head+"\n".join(ev)+"\n")
r=subprocess.run(["ffmpeg","-y","-i",a.video,"-vf",f"ass={ass}","-c:v","libx264","-preset","medium",
    "-crf","18","-pix_fmt","yuv420p","-c:a","copy",a.out],capture_output=True,text=True)
print("烧录 rc=",r.returncode, "" if r.returncode==0 else r.stderr[-400:])
if r.returncode==0: print(f"完成 -> {a.out} ({os.path.getsize(a.out)//1000000}MB)")
