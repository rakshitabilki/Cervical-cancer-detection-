import os
import shutil

base_path = "dataset"

for category in ["normal", "cancer"]:
    category_path = os.path.join(base_path, category)

    for folder in os.listdir(category_path):
        folder_path = os.path.join(category_path, folder)

        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                src = os.path.join(folder_path, file)
                dst = os.path.join(category_path, file)

                shutil.move(src, dst)

            os.rmdir(folder_path)

print("✅ Images moved successfully!")