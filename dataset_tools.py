import os
import shutil
import random

TEST_PATH = "./asl_alphabet_test/asl_alphabet_test"
TRAIN_PATH = "./asl_alphabet_train/asl_alphabet_train"
NUM_IMAGES_TO_MOVE = 600

def remove_postfix_from_files(test_path, postfix="_test"):
    # Walk through all subdirectories and files
    for root, dirs, files in os.walk(test_path):
        for file in files:
            # Check if the file name contains the postfix
            if postfix in file:
                # Create the new file name by removing the postfix
                new_file_name = file.replace(postfix, '')
                # Rename the file
                os.rename(os.path.join(root, file), os.path.join(root, new_file_name))

def move_images(src_dir, dest_dir, num_images):
    # Get a list of all files in the source directory
    files = os.listdir(src_dir)
    # Select a random sample of the specified number of images
    files_to_move = random.sample(files, num_images)
    for file in files_to_move:
        # Construct full file path
        src_file = os.path.join(src_dir, file)
        dest_file = os.path.join(dest_dir, file)
        # Move the file
        shutil.move(src_file, dest_file)

def move_images_for_all_subdirectories(train_path, test_path, num_images):
    # Get a list of all subdirectories in the training directory
    subdirectories = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
    for subdirectory in subdirectories:
        src_dir = os.path.join(train_path, subdirectory)
        dest_dir = os.path.join(test_path, subdirectory)
        # Ensure the destination directory exists
        os.makedirs(dest_dir, exist_ok=True)
        # Move images from the current subdirectory
        move_images(src_dir, dest_dir, num_images)

# Run the function to remove postfix from files
# remove_postfix_from_files(TEST_PATH)

# Run the function to move images for all subdirectories
# move_images_for_all_subdirectories(TRAIN_PATH, TEST_PATH, NUM_IMAGES_TO_MOVE)