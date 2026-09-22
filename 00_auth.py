"""
Run once. Sets Kaggle credentials.
Preferred: place kaggle.json at ~/.kaggle/kaggle.json (Linux/Mac)
or C:\\Users\\<you>\\.kaggle\\kaggle.json (Windows) — download it from
kaggle.com/settings -> Create New Token.

If you'd rather not manage that file, uncomment below and hardcode
(then don't commit this file with real keys to git).
"""
import os
os.environ['KAGGLE_USERNAME'] = 'Aracknab69'
os.environ['KAGGLE_KEY'] = 'KGAT_7d5a7d82145a571c31d09580f32c1658'

print("If kaggle.json is in place, you're done. Run 01_download.py next.")