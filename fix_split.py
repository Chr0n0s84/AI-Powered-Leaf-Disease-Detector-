import os, glob, shutil, random

classes = ['healthy', 'bacterial_spot', 'early_blight', 'late_blight', 'leaf_mold', 'powdery_mildew']
src_dir = 'dataset'
dst_dir = 'dataset_clean'

if os.path.exists(dst_dir):
    shutil.rmtree(dst_dir)

for c in classes:
    # Gather all images across any existing subfolders
    imgs = glob.glob(f'{src_dir}/**/{c}/*.*', recursive=True)
    imgs = [f for f in imgs if not f.endswith('.DS_Store')]
    
    random.shuffle(imgs)
    
    os.makedirs(f'{dst_dir}/train/{c}', exist_ok=True)
    os.makedirs(f'{dst_dir}/val/{c}', exist_ok=True)
    
    split = int(len(imgs) * 0.8)
    for idx, f in enumerate(imgs[:split]):
        shutil.copy(f, f'{dst_dir}/train/{c}/img_{idx}.jpg')
    for idx, f in enumerate(imgs[split:]):
        shutil.copy(f, f'{dst_dir}/val/{c}/img_{idx}.jpg')

# Swap folders
shutil.rmtree(src_dir)
os.rename(dst_dir, src_dir)
print("Dataset completely rebuilt and aligned successfully!")
