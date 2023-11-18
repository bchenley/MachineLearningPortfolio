def get_images(image_path, dtype=np.float32):
  # Load the NIfTI file using nibabel
  images = nib.load(image_path)

  # Access the data from the NIfTI file and convert it to the specified data type
  data = images.get_fdata().astype(dtype)

  # Return the image data array
  return data
