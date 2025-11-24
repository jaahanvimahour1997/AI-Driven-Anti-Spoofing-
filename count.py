import os

# Change this to your dataset folder
DATASET_PATH = r"C:\Users\hp\PycharmProjects\PythonProject21\dataset"

real_count = 0
spoof_count = 0

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):

            full_path = os.path.join(root, file)

            # Check folder name
            if "real" in root.lower():
                real_count += 1
            elif "spoof" in root.lower():
                spoof_count += 1

print("------------ IMAGE COUNT ------------")
print(f"Real Images   : {real_count}")
print(f"Spoof Images  : {spoof_count}")
print("-------------------------------------")
