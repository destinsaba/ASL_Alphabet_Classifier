import os

TEST_PATH = "./asl_alphabet_test/asl_alphabet_test"

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

# Run the function
remove_postfix_from_files(TEST_PATH)