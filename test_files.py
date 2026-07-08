from google import genai
from google.genai import types as gtypes
import tempfile, os

c = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

# Create tiny test file
tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
tmp.write(b'\x00' * 1024)
tmp.close()

try:
    with open(tmp.name, 'rb') as f:
        ref = c.files.upload(file=f, config=gtypes.UploadFileConfig(mime_type='video/mp4'))
    print('Files API OK:', ref.name)
    c.files.delete(name=ref.name)
except Exception as e:
    print('Files API ERROR:', e)
finally:
    os.unlink(tmp.name)
