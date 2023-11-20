import SimpleITK as sitk
import numpy as np

def register_subject_images(images,
                            fixed_image_idx = 0,
                            normalize_images = False,
                            transform_type = 'similarity',
                            metric = 'mmi',
                            num_bins = 50,
                            learning_rate = 1.0,
                            max_iters = 200,
                            interp_method = 'linear',
                            default_pixel_value = 0.,
                            min_convergence = 1e-6, convergence_window = 20,
                            sitk_dtype = sitk.sitkFloat32):

  num_subjects = len(images)
  num_images = images[0]['images'].shape[-1]

  moving_image_idx = np.where(np.arange(num_images) != fixed_image_idx)[0]

  for i,subject_images in enumerate(images):
    subject_images['scaler'] = None
    if normalize_images:
      subject_images['scaler'] = []
      for i in range(num_images):
        scaler = MinMaxScaler(feature_range = (0, 1))

        image_i = subject_images['image'][:, :, :, i]
        
        subject_images['images'][:, :, :, i] = scaler.fit_transform(image_i.reshape(-1, 1)).reshape(image_i.shape)
        subject_images['scaler'].append(scaler)

    fixed_image = subject_images['images'][:, :, :, fixed_image_idx]

    for idx in moving_image_idx:
      print(f"Registering image {idx+1} to image {fixed_image_idx+1} for subject {subject_images['id']} ({i+1}/{num_subjects}))")

      subject_images['images'][:, :, :, idx], _ = register_image(fixed_image = fixed_image,
                                                                  moving_image = subject_images['images'][:, :, :, idx],
                                                                  transform_type = transform_type,
                                                                  metric = metric,
                                                                  num_bins = num_bins,
                                                                  learning_rate = learning_rate,
                                                                  max_iters = max_iters,
                                                                  interp_method = interp_method,
                                                                  default_pixel_value = default_pixel_value,
                                                                  min_convergence = min_convergence,
                                                                  convergence_window = convergence_window,
                                                                  sitk_dtype = sitk_dtype)

  return images
