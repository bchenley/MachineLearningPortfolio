def access_images(task_path,
                  image_dir='imagesTr/',
                  label_dir='labelsTr/',
                  dtype=None,
                  sample_size=None, random_seed=42,
                  register_images=False,
                  fixed_image_idx=0,
                  normalize_images=False,
                  transform_type='similarity',
                  metric='mmi',
                  num_bins=50,
                  learning_rate=1.0,
                  max_iters=200,
                  interp_method='linear',
                  default_pixel_value=0.,
                  min_convergence=1e-6, convergence_window=20,
                  sitk_dtype=sitk.sitkFloat32):

    # Construct the dataset and label paths from the given directories
    dataset_path = os.path.join(task_path, image_dir)
    label_path = None
    if 'Tr' in image_dir:
        label_path = os.path.join(task_path, label_dir)

    # List all NIfTI files in the dataset_path that match the criteria
    nii_list = [nii for nii in os.listdir(dataset_path) if ('nii' in nii) and nii.startswith('BRATS')]

    # Sample a subset of files if a sample size is specified
    if sample_size is not None:
        if random_seed is not None: np.random.seed(random_seed)
        np.random.shuffle(nii_list)
        nii_list = nii_list[:sample_size]

    # Get the total number of subjects to process
    num_subjects = len(nii_list)

    # Initialize an empty list to hold the data
    data = []
    for i, image_file in enumerate(tqdm(nii_list, desc="Loading NIfTI files", unit='file')):

        # Extract the ID from the filename
        id = image_file.split('_')[1].split('.')[0]

        # Append a new dictionary to hold data for the current subject
        data.append({})

        # Assign the extracted ID to the current subject's data
        data[-1]['id'] = id

        # Retrieve the image data using the get_images function
        data[-1]['images'] = get_images(os.path.join(dataset_path, image_file))

        # Register images if specified
        if register_images:
            # Get the number of images (e.g., time points or modalities)
            num_images = data[-1]['images'].shape[-1]

            # Determine the indices of images that are not the fixed image
            moving_image_idx = np.where(np.arange(num_images) != fixed_image_idx)[0]

            # Initialize the image scaler if image normalization is needed
            data[-1]['image_scaler'] = None
            if normalize_images:
                data[-1]['image_scaler'] = []
                for j in range(num_images):
                    # Initialize the MinMaxScaler
                    scaler = MinMaxScaler(feature_range=(0, 1))

                    # Retrieve the j-th image from the current subject
                    image_j = data[-1]['images'][:, :, :, j]
                    if isinstance(image_j, torch.Tensor):
                        # Normalize the tensor image and retain its type and device
                        image_js = torch.tensor(scaler.fit_transform(image_j.cpu().reshape(-1, 1))
                                                .reshape(image_j.shape)).to(image_j.device, image_j.dtype)
                    else:
                        # Normalize the numpy array image
                        image_js = scaler.fit_transform(image_j.reshape(-1, 1)).reshape(image_j.shape)

                    # Update the j-th image with the normalized image
                    data[-1]['images'][:, :, :, j] = image_js
                    # Append the scaler to the list of scalers for future inverse transformation if necessary
                    data[-1]['image_scaler'].append(scaler)

            # Set the fixed image for registration
            fixed_image = data[-1]['images'][:, :, :, fixed_image_idx]

            # Register each moving image to the fixed image
            for idx in moving_image_idx:
                print()
                print(f"Registering image {idx+1} to image {fixed_image_idx+1} for subject {data[-1]['id']} ({i+1}/{num_subjects}))")
                data[-1]['images'][:, :, :, idx], _ = register_image(fixed_image=fixed_image,
                                                                     moving_image=data[-1]['images'][:, :, :, idx],
                                                                     transform_type=transform_type,
                                                                     metric=metric,
                                                                     num_bins=num_bins,
                                                                     learning_rate=learning_rate,
                                                                     max_iters=max_iters,
                                                                     interp_method=interp_method,
                                                                     default_pixel_value=default_pixel_value,
                                                                     min_convergence=min_convergence,
                                                                     convergence_window=convergence_window,
                                                                     sitk_dtype=sitk_dtype)
        
        # If a label path is specified, load the labels for the current subject
        if label_path is not None:
            labels = get_images(os.path.join(label_path, image_file))
            data[-1]['labels'] = labels

    # Return the list of subjects with their corresponding images and labels
    return data
