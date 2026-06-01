import os
import sys
import importlib.util
from PIL import Image

root = os.path.dirname(__file__)
nst_code_dir = os.path.join(root, 'NST_Code')
# Ensure NST_Code is first on sys.path so `from utils...` in app.py resolves to NST_Code/utils
if nst_code_dir not in sys.path:
	sys.path.insert(0, nst_code_dir)

app_path = os.path.join(nst_code_dir, 'app.py')
spec = importlib.util.spec_from_file_location('nst_app', app_path)
nst_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nst_app)

content_img = Image.open(os.path.join(nst_code_dir, 'examples', 'brad_pitt.jpg')).convert('RGB')
style_img = Image.open(os.path.join(nst_code_dir, 'examples', 'picasso_seated_nude_hr.jpg')).convert('RGB')

print('Running style transfer...')
result = nst_app.style_transfer(content_img, style_img, nst_app.encoder, nst_app.decoder, alpha=1.0, device=nst_app.device)

out_dir = os.path.join(root, 'static', 'uploads')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'sample_stylized.jpg')

nst_app.save_image(result, out_path)
print('Saved:', out_path)
