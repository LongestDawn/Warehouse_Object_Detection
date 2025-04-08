#This script splits between train and Validation folders for training the YOLO11 model

from pathlib import Path
import random
import shutil
import argparse

#Creates the directory (File structure) that Ultralytics requires (The train and validation datasets)
def make_dirs(paths):
    for path in paths:
        #Create each directory in list.
        path.mkdir(parents=True, exist_ok=True)

#Split a list of image paths into training and validation sets (This is to be based on a ratio)
def split_dataset(image_paths, train_ratio):
    random.shuffle(image_paths) #This randomizes the image list
    train_count = int(len(image_paths) * train_ratio) #This calculates the number of training images
    return image_paths[:train_count], image_paths[train_count:] #Splits the dataset

#Copies the, "Image" and, "Label" folders from the original to the destination folders
def copy_files(file_list, label_dir, dst_img_dir, dst_lbl_dir): 
    for img_path in file_list:
        #Copy the, "Image" file to the destination image directory
        shutil.copy(img_path, dst_img_dir / img_path.name)

        #Creates the corresponding label (.txt) file path
        label_path = label_dir / f"{img_path.stem}.txt"

        #Copies the, "Label" file (If it exists)
        if label_path.exists():
            shutil.copy(label_path, dst_lbl_dir / label_path.name)

#This does the main function
def main():
    #Parser command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', required=True, help='Path to dataset containing images and labels folders')
    parser.add_argument('--train_pct', type=float, default=0.8, help='Percentace of data to use for the training (0.01 - 0.99)')
    args = parser.parse_args()

    #Defines the input folder paths for the, "Image" and "Label" folders
    data_path = Path(args.datapath)
    train_ratio = args.train_pct

    image_dir = data_path / 'images'
    label_dir = data_path / 'labels'

    #Defines the output folder paths for the ,"train" and "validation" splits
    output_root = Path.cwd() / 'data'
    train_img_dir = output_root / 'train' / 'images'
    train_lbl_dir = output_root / 'train' / 'labels'
    val_img_dir = output_root / 'validation' / 'images'
    val_lbl_dir = output_root / 'validation' / 'labels'
    
    #Creates all of the needed output directories
    make_dirs([train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir])

    #Gets all of the image file paths (Except for the hidden/ system files)
    image_paths = [p for p in image_dir.rglob('*') if p.is_file() and not p.name.startswith('.')]
    print(f"Total images: {len(image_paths)}")

    #Splits the image files to the Training and Validation sets
    train_set, val_set = split_dataset(image_paths, train_ratio)
    print(f"Train: {len(train_set)}")
    print(f"Validation: {len(val_set)}")

    #Copies the training images and labels to the output train folder
    copy_files(train_set, label_dir, train_img_dir, train_lbl_dir)

    #Copies the validation images and labels to the output validation folder
    copy_files(val_set, label_dir, val_img_dir, val_lbl_dir)

#Runs the Script
if __name__ == '__main__':
    main()