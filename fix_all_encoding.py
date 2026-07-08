# Fix all garbled Chinese strings in app.py caused by PowerShell encoding corruption
fixes = {
    # Job status messages
    'f"瀹屾垚锛佸叡鐢熸垚 {len(subtitles)} 鏉″瓧骞?': 'f"完成！共生成 {len(subtitles)} 条字幕"',
    'f"瀛楀箷鐢熸垚瀹屾垚锛屽叡 {len(subtitles)} 鏉?': 'f"字幕生成完成，共 {len(subtitles)} 条"',
    'f"澶勭悊澶辫触锛歿str(e)}"': 'f"处理失败：{str(e)}"',
    'f"閿欒锛歿str(e)}"': 'f"错误：{str(e)}"',
    '"宸蹭笂浼犮€佸紑濮嬪鐞嗭紙.."': '"已上传，开始处理..."',
    '"姝e湪涓婁紶瑙嗛鍒?Gemini..."': '"正在上传视频到 Gemini..."',
    '"Gemini 澶勭悊涓?.."': '"Gemini 处理中..."',
    '"Gemini 鍒嗘瀽瑙嗛涓?.."': '"Gemini 分析视频中..."',
    '"AI 鐢熸垚瀛楀箷涓?.."': '"AI 生成字幕中..."',
    '"AI 鐢熸垚涓枃瀛楀箷..."': '"AI 生成中文字幕..."',
    '"閲嶆柊澶勭悊涓?.."': '"重新处理中..."',
    '"鍒濆鍖栭厤闊筹紙.."': '"初始化配音..."',
    'f"鍚堟垚绗?{i+1}/{len(subtitles)} 鏉″瓧骞?': 'f"合成第 {i+1}/{len(subtitles)} 条字幕..."',
    'f"TTS閿欒(绗?{i+1}鏉?): {str(e)[:80]}"': 'f"TTS错误(第{i+1}条): {str(e)[:80]}"',
    'f"鍏ㄥ眬璇熷害: {speed:.2f}x"': 'f"全局语速: {speed:.2f}x"',
    '"璁$畻璇熷害..."': '"计算语速..."',
    '"鍚堝苟闊抽鏈€橀亾..."': '"合并音频轨道..."',
    '"鍚堝苟鍒拌棰戜腑..."': '"合并到视频中..."',
    '"閰嶉煶瀹屾垚锛?"': '"配音完成！"',
    'f"閰嶉煶澶辫触锛歿str(e)}"': 'f"配音失败：{str(e)}"',
    '"鍒濆鍖栭厤闊?.."': '"初始化配音..."',
    '"鍑嗗瀛楀箷鏂囦欢..."': '"准备字幕文件..."',
    '"鐑ч褰曞瓧骞?..': '"烧录字幕..."',
    '"瀹屾垚锛?"': '"完成！"',
    'f"澶辫触锛歿str(e)}"': 'f"失败：{str(e)}"',
    '"鍒濆鍖?..': '"初始化..."',
    '"杞崲鏂囦欢涓哄浘鐗?..': '"转换文件为图片..."',
    'f"鍏? {len(page_images)} 椤碉紝鐢熸垚璁叉瑙i锛?"': 'f"共 {len(page_images)} 页，生成讲解词..."',
    'f"Gemini 鐢熸垚绗?{i+1}/{len(page_images)} 椤佃璁层€?': 'f"Gemini 生成第 {i+1}/{len(page_images)} 页讲解词..."',
    'f"鍚堟垚绗?{i+1}/{len(page_images)} 椤侀厤闊筹紙.."': 'f"合成第 {i+1}/{len(page_images)} 页配音..."',
    '"鍚堝苟闊抽..."': '"合并音频..."',
    '"鐢熸垚瑙嗛..."': '"生成视频..."',
    'f"瀹屾垚锛佸叡 {len(page_images)} 椤?"': 'f"完成！共 {len(page_images)} 页"',
    'f"鏂囦欢涓嶆敮鎸乊锛歿suffix}"': 'f"不支持的文件格式：{suffix}"',
    '"PDF 杞崲澶辫触锛屾病鏈夌敓鎴愬浘鐗?"': '"PDF 转换失败，没有生成图片"',
    '"娌℃湁鎵惧埌瀛楀箷鏁版嵁"': '"没有找到字幕数据"',
    '"浠诲姟涓嶅瓨鍦?"': '"任务不存在"',
    '"瀛楀箷鐑ч褰曞畬鎴愶紙.."': '"字幕烧录完成！"',
    '"涓嬪畬鎴?"': '"完成！"',
    '"閰嶉煶涓?.."': '"配音中..."',
    'f"鐑ч褰曞嚭鏉?"': 'f"烧录出来了"',
    # Log messages  
    '"涓婁紶瑙嗛鍒?Gemini Files API..."': '"上传视频到 Gemini Files API..."',
    '"Gemini 鍒嗘瀽瑙嗛涓?.."': '"Gemini 分析视频中..."',
    # status strings
    '"uploading_gemini"': '"uploading_gemini"',
    # Project title in print
    'Video Subtitle Generator starting on': 'Video Subtitle Generator starting on',
    # Error messages  
    '"鏈缃?GOOGLE_API_KEY"': '"未设置 GOOGLE_API_KEY"',
    '"娌℃湁涓婁紶鏂囦欢"': '"没有上传文件"',
    '"浠诲姟涓嶅瓨鍦?"': '"任务不存在"',
    '"video not found"': '"video not found"',
    # Subtitle messages
    f'"宸茬粡鏄樊涓?.."': '"已经是默认了..."',
    # Add/Log messages
    '"浠诲姟涓嶅瓨鍦?"': '"任务不存在"',
    '"澶嶅埗锛?copy"': '"copy"',
    # Keep non-Chinese strings as is
}

content = open('C:/Users/tanga/video-subtitle/app.py', encoding='utf-8', errors='replace').read()

fixed_count = 0
for garbled, correct in fixes.items():
    if garbled in content:
        content = content.replace(garbled, correct)
        fixed_count += 1
        print(f"Fixed: {correct[:50]}")

# Try to detect remaining garbled text (common garbled CJK patterns)
import re
# Look for lines with mojibake patterns (valid Chinese that shouldn't be there due to double-encoding)
garbled_lines = []
for i, line in enumerate(content.split('\n'), 1):
    # Check for garbled sequences - characters in CJK Unified Ideographs that look like mojibake
    # Mojibake from UTF-8 read as GBK typically produces chars in 0x9000-0x9fff range mixed with others
    if re.search(r'[鍏鏉鐢熺鈥?\u9780-\u9fff]{3,}', line) and ('"' in line or "'" in line):
        garbled_lines.append((i, line.strip()[:80]))

if garbled_lines:
    print(f"\nWARNING: {len(garbled_lines)} lines may still have garbled text:")
    for lineno, line in garbled_lines[:15]:
        print(f"  Line {lineno}: {line}")
else:
    print("\nNo remaining garbled text detected")

print(f"\nTotal fixed: {fixed_count}")
open('C:/Users/tanga/video-subtitle/app.py', 'w', encoding='utf-8').write(content)
