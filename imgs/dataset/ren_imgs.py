import os

parent_folder_path = "./imgs/ref_imgs/original/test"

subfolder_list = os.listdir(parent_folder_path)

for folder in subfolder_list:
    new_prefix = folder.lower()

    folder_path = os.path.join(parent_folder_path, folder)
    file_list = os.listdir(folder_path)

    file_list.sort()

    for index, old_name in enumerate(file_list, start=1):
        _, file_extension = os.path.splitext(old_name)

        new_name = f"{new_prefix}{index:04d}{file_extension}"

        # Construct full paths for the old and new names
        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)

        # Rename the file
        os.rename(old_path, new_path)

        print(f"Renamed {old_name} to {new_name}")

print("Renaming complete.")
