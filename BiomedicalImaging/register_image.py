def register_image(fixed_image, moving_image,
                   transform_type = 'similarity',
                   metric = 'mmi',
                   num_bins = 50,
                   learning_rate = 1.0,
                   max_iters = 200,
                   interp_method = 'linear',
                   default_pixel_value = 0.,
                   min_convergence = 1e-6, convergence_window = 20,
                   sitk_dtype = sitk.sitkFloat32):

  device, final_dtype = None, None
  # Get fixed image as sitk array
  if isinstance(fixed_image, np.ndarray):
    if fixed_image.dtype == np.float16:
      final_dtype = np.float16
    elif fixed_image.dtype == np.float32:
      final_dtype = np.float32
    elif fixed_image.dtype == np.float64:
      final_dtype = np.float64

    fixed_image = sitk.GetImageFromArray(fixed_image)

  # Get moving image as sitk array
  if isinstance(moving_image, str):
    moving_image = sitk.ReadImage(moving_image, sitk_dtype)
  elif isinstance(moving_image, np.ndarray):
    moving_image = sitk.GetImageFromArray(moving_image)
  elif isinstance(moving_image, torch.Tensor):
    moving_image = sitk.GetImageFromArray(moving_image.cpu())

  assert fixed_image.GetSize() == moving_image.GetSize(), "Image sizes do not match"
  assert fixed_image.GetSpacing() == moving_image.GetSpacing(), "Image spacings do not match"
  assert fixed_image.GetDimension() == moving_image.GetDimension(), "Image dimensions do not match"

  # Instantiate registration method
  R = sitk.ImageRegistrationMethod()

  # Set up metric
  if metric == 'mmi':
    R.SetMetricAsMattesMutualInformation(num_bins)
  elif metric == 'ms':
    R.SetMetricAsMeanSquares()
  elif metric == 'jhmi':
    R.SetMetricAsJointHistogramMutualInformation(num_bins)
  else:
    raise ValueError(f"The metric ({metric}) must be 'mmi', 'ms', or 'jhmi'.")

  # Set optimizer
  R.SetOptimizerAsGradientDescent(learningRate = learning_rate,
                                  numberOfIterations = max_iters,
                                  convergenceMinimumValue = min_convergence,
                                  convergenceWindowSize = convergence_window)

  # Define transform
  if transform_type == 'similarity':
    transform = sitk.Similarity3DTransform()
  elif transform_type == 'euler':
    transform = sitk.Euler3DTransform()
  elif transform_type == 'translation':
    transform = sitk.TranslationTransform()
  elif transform_type == 'versor':
    transform = sitk.VersorTransform()
  elif transform_type == 'versorrigid':
    transform = sitk.VersorRigid3DTransform()
  elif transform_type == 'scale':
    transform = sitk.ScaleTransform()
  elif transform_type == 'scaleversor':
    transform = sitk.ScaleVersor3DTransform()
  elif transform_type == 'scaleskewversor':
    transform = sitk.ScaleSkewVersor3DTransform()
  elif transform_type == 'composescaleskewversor':
    transform = sitk.ComposeScaleSkewVersor3DTransform()
  elif transform_type == 'affine':
    transform = sitk.AffineTransform()
  elif transform_type == 'bspline':
    transform = sitk.BSplineTransform()
  elif transform_type == 'displacement':
    transform = sitk.DisplacementFieldTransform()
  elif transform_type == 'composite':
    transform = sitk.CompositeTransform()

  # Align centers of fixed and moving image
  initial_transform = sitk.CenteredTransformInitializer(fixedImage = fixed_image,
                                                        movingImage = moving_image,
                                                        transform = transform)

  R.SetInitialTransform(initial_transform)

  # Set scales for optimization
  R.SetOptimizerScalesFromIndexShift()

  # Set up interpolator
  if interp_method == 'linear':
    interpolator = sitk.sitkLinear
  elif interp_method == 'nn':
    interpolator = sitk.sitkNearestNeighbor
  elif interp_method == 'spline':
    interpolator = sitk.sitkBSpline
  else:
    raise ValueError(f"The interpolator ({interpolator}) must be 'linear', 'nn', or 'spline'.")

  R.SetInterpolator(interpolator)

  # Set progress
  metric_values = []

  def command_iteration(method):

    metric_values.append(method.GetMetricValue())

    print(f"Iter. {method.GetOptimizerIteration():3} "
          + f"= {metric_values[-1]:.4f} ")
          # + f": {method.GetOptimizerPosition()}")

  R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

  # Set transform
  final_transform = R.Execute(fixed = fixed_image, moving = moving_image)

  print(final_transform)
  print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
  print(f"Iteration: {R.GetOptimizerIteration()}")
  print(f"Metric: {R.GetMetricValue():.4f}")

  # Apply transformation to image and interpolate the result
  resampler = sitk.ResampleImageFilter()
  resampler.SetReferenceImage(fixed_image)
  resampler.SetInterpolator(interpolator)
  resampler.SetDefaultPixelValue(default_pixel_value)
  resampler.SetTransform(final_transform)

  output_image = resampler.Execute(moving_image)
  
  # Convert moving image to original dtype
  if final_dtype in [np.float16, np.float32, np.float64]:
    output_image = sitk.GetArrayFromImage(output_image).astype(final_dtype)

  return output_image, metric_values
