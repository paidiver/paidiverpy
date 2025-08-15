def process_sequentially(self, images: xr.Dataset, method: callable, params: dict, custom: bool = False) -> xr.Dataset:
    """
    Process the last step's images sequentially and append results as a new step.

    Args:
        images (xr.Dataset): Dataset containing 'image' variable and metadata as coordinates.
        method (callable): The method to apply to the images.
        params (dict): The parameters for the method.
        custom (bool, optional): Whether the method is a custom method. Defaults to False.

    Returns:
        xr.Dataset: Updated dataset with a new step containing processed images.
    """
    func = partial(method, params=params)

    # Get the last step value
    last_step_val = images.step.max().item()
    last_step_images = images["image"].sel(step=last_step_val)

    # Metadata for processing
    metadata = self.get_metadata(orient="records")

    # Process with apply_ufunc
    processed_images, updated_metadata_list = xr.apply_ufunc(
        Paidiverpy.process_single,
        last_step_images,
        metadata,
        input_core_dims=[["y", "x", "band"]],
        output_core_dims=[["y", "x", "band"], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[images["image"].dtype, object],
    )

    # Determine the new step value
    new_step_val = last_step_val + 1

    # Expand dims for new step and concat
    processed_images = processed_images.expand_dims(step=[new_step_val])
    new_ds = xr.concat([images, images.assign(image=processed_images)], dim="step")

    # Update metadata coordinates for filename
    for coord in updated_metadata_list[0].keys():
        new_vals = [meta[coord] for meta in updated_metadata_list]
        new_ds[coord].loc[dict(filename=new_ds.filename)] = new_vals

    return new_ds
