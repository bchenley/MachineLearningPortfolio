## Import modules
import os, random, pickle

## import access_images
from src.access_images import access_images

# Get user input for a given prompt
def get_user_input(prompt):
    return input(prompt)

# Validate user input
def validate_input(input_text, prompt):
    while True:
        user_input = get_user_input(prompt)
        confirm = input(f"Confirm {input_text} (y/n): ").lower()
        if confirm == 'y':
            return user_input

# Initialize variables
task_path = ""
save_dir = ""
sample_size = 100
train_size = 0
test_size = 0
register_images = False

# '/content/drive/MyDrive/data/MSD/Task01_BrainTumour'

# Get user input and validate
task_path = validate_input("path to DICOM dataset", "Enter path to DICOM dataset:")
save_dir = validate_input("path to save images", "Enter path to save images:")
sample_size = int(validate_input("sample size (default = 100)", "Enter sample size (default = 100):"))
train_size = int(sample_size * float(validate_input("% training size [0, 1]", "Enter % training size [0, 1]:")))
test_size = sample_size - train_size
register_images_input = validate_input("register Images to T1 (y/n)", "Register Images to T1 (y/n)?:")
register_images = True if register_images_input == 'y' else False

# Display user input for confirmation
print(f"------------------------------------")
print(f"Path to DICOM dataset = {task_path}")
print(f"Path to save images = {save_dir}")
print(f"Desired Sample size = {sample_size}")
print(f"Training size = {train_size}")
print(f"Test size = {test_size}")
print(f"T1 Image Registration {'enabled.' if register_images else 'disabled.'}")

## Get images, performing registration if desired

images = access_images(task_path = task_path,
                       sample_size = sample_size,
                       register_images = register_images)


## Split images to train and test sets

random.shuffle(images)

train_images = images[:train_size]
test_images = images[test_size:]

## Save training and test images

train_path = f"{save_dir}/train.pkl"
with open(train_path, "wb") as file:
  pickle.dump(train_images, file)

print("Training images successfully saved.")

test_path = f"{save_dir}/test.pkl"
with open(test_path, "wb") as file:
  pickle.dump(test_images, file)

print("Test images successfully saved.")  
