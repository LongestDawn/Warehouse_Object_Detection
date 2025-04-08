from pathlib import Path
import random
import shutil
import argparse

def make_dirs(paths):
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

def split_dataset(image_paths, train_ratio):
    random.shuffle(image_paths)
    train_count = int(len(image_paths) * train_ratio)
    return image_paths[:train_count], image_paths[train_count:]

def copy_files(file_list, label_dir, dst_img_dir, dst_lbl_dir):
    for img_path in file_list:
        shutil.copy(img_path, dst_img_dir / img_path.name)
        label_path = label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy(label_path, dst_lbl_dir / label_path.name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', required=True)
    parser.add_argument('--train_pct', type=float, default=0.8)
    args = parser.parse_args()

    data_path = Path(args.datapath)
    train_ratio = args.train_pct

    image_dir = data_path / 'images'
    label_dir = data_path / 'labels'

    output_root = Path.cwd() / 'data'
    train_img_dir = output_root / 'train' / 'images'
    train_lbl_dir = output_root / 'train' / 'labels'
    val_img_dir = output_root / 'validation' / 'images'
    val_lbl_dir = output_root / 'validation' / 'labels'
    
    make_dirs([train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir])

    image_paths = [p for p in image_dir.rglob('*') if p.is_file() and not p.name.startswith('.')]
    print(f"Total images: {len(image_paths)}")

    train_set, val_set = split_dataset(image_paths, train_ratio)
    print(f"Train: {len(train_set)} | Validation: {len(val_set)}")

    copy_files(train_set, label_dir, train_img_dir, train_lbl_dir)
    copy_files(val_set, label_dir, val_img_dir, val_lbl_dir)

if __name__ == '__main__':
    main()