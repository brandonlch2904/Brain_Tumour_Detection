import cv2
import os
import matplotlib.pyplot as plt

# Define paths
input_base = "../data/raw"  # Raw MRI images
output_base_resize_first = "../data/preprocessed/resize_then_enhance"
output_base_enhance_first = "../data/preprocessed/enhance_then_resize"

# Ensure output directories exist
for dataset_type in ["Training", "Testing"]:
    for category in ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]:
        os.makedirs(os.path.join(output_base_resize_first, dataset_type, category), exist_ok=True)
        os.makedirs(os.path.join(output_base_enhance_first, dataset_type, category), exist_ok=True)

print("\n✅ Folder structure for preprocessed data is set up successfully.")

#=================================#
# Noise Removal & Contrast Enhancement #
#=================================#

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

def preprocess_image(image):
    """Applies noise removal and contrast enhancement."""
    img_denoised = cv2.GaussianBlur(image, (3, 3), 0)
    img_enhanced = clahe.apply(img_denoised)
    return img_enhanced

#=================================#
# Resizing Methods #
#=================================#

def resize_then_enhance(image):
    img_resized = cv2.resize(image, (224, 224))
    return preprocess_image(img_resized)

def enhance_then_resize(image):
    img_enhanced = preprocess_image(image)
    return cv2.resize(img_enhanced, (224, 224))

#========================#
# Process Entire Dataset #
#========================#

# print("\n🧠 Starting Preprocessing and Resizing...")

# for dataset_type in ["Training", "Testing"]:
#     for category in ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]:
#         input_folder = os.path.join(input_base, dataset_type, category)
#         output_folder_resize_first = os.path.join(output_base_resize_first, dataset_type, category)
#         output_folder_enhance_first = os.path.join(output_base_enhance_first, dataset_type, category)

#         image_list = os.listdir(input_folder)
#         total_images = len(image_list)
#         print(f"\nProcessing {dataset_type} -> {category} ({total_images} images)")

#         for idx, filename in enumerate(image_list):
#             input_path = os.path.join(input_folder, filename)

#             img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
#             img_resize_first = resize_then_enhance(img)
#             img_enhance_first = enhance_then_resize(img)

#             cv2.imwrite(os.path.join(output_folder_resize_first, filename), img_resize_first)
#             cv2.imwrite(os.path.join(output_folder_enhance_first, filename), img_enhance_first)

#             if (idx + 1) % 50 == 0 or idx == total_images - 1:
#                 print(f"  Processed {idx + 1}/{total_images} images...")

# print("\n✅ Preprocessing and Resizing Completed!")

#====================#
# Visualize a Sample #
#====================#

sample_path = os.path.join(input_base, "Training/glioma_tumor")
sample_image_file = os.listdir(sample_path)[0]
sample_image_path = os.path.join(sample_path, sample_image_file)

sample_img = cv2.imread(sample_image_path, cv2.IMREAD_GRAYSCALE)
sample_resize_first = resize_then_enhance(sample_img)
sample_enhance_first = enhance_then_resize(sample_img)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(sample_img, cmap='gray')
plt.title("Original MRI Scan")

plt.subplot(1, 3, 2)
plt.imshow(sample_resize_first, cmap='gray')
plt.title("Resize → Enhance")

plt.subplot(1, 3, 3)
plt.imshow(sample_enhance_first, cmap='gray')
plt.title("Enhance → Resize")

plt.show()
