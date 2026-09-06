import os
import glob
import shutil
import random

classes = ['healthy', 'bacterial_spot', 'early_blight', 'late_blight', 'leaf_mold', 'powdery_mildew']

for c in classes:
    # Get all images for this class
    train_imgs = glob.glob(f'dataset/train/{c}/*.*')
    val_imgs = glob.glob(f'dataset/val/{c}/*.*')
    all_imgs = train_imgs + val_imgs

    if not all_imgs:
        continue

    random.shuffle(all_imgs)

    # Move images to a temporary backup directory with unique names
    temp_dir = f'dataset/temp_{c}'
    os.makedirs(temp_dir, exist_ok=True)
    for idx, img in enumerate(all_imgs):
        ext = os.path.splitext(img)[1]
        dest = os.path.join(temp_dir, f'img_{idx}{ext}')
        shutil.move(img, dest)

    # Re-create clean target directories
    shutil.rmtree(f'dataset/train/{c}', ignore_errors=True)
    shutil.rmtree(f'dataset/val/{c}', ignore_errors=True)
    os.makedirs(f'dataset/train/{c}', exist_ok=True)
    os.makedirs(f'dataset/val/{c}', exist_ok=True)

    # Move files back in an 80/20 train/val ratio
    moved_files = glob.glob(f'{temp_dir}/*.*')
    split_idx = int(len(moved_files) * 0.8)

    for f in moved_files[:split_idx]:
        shutil.move(f, f'dataset/train/{c}/')
    for f in moved_files[split_idx:]:
        shutil.move(f, f'dataset/val/{c}/')

    # Remove temporary directory
    shutil.rmtree(temp_dir)

print("Dataset successfully rebalanced to 80% train / 20% val!")
