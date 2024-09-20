""" Open raw image file
"""

import dask.delayed
import numpy as np
import cv2
from tqdm import tqdm

import dask
from dask import delayed, compute
from dask.diagnostics import ProgressBar
import dask.array as da
from paidiverpy import Paidiverpy
from paidiverpy.config.convert_params import (
    CONVERT_LAYER_METHODS,
    BitParams,
    ToParams,
    BayerPatternParams,
    NormalizeParams,
    ResizeParams,
    CropParams,
)


class ConvertLayer(Paidiverpy):
    def __init__(
        self,
        config_file_path=None,
        input_path=None,
        output_path=None,
        catalog_path=None,
        catalog_type=None,
        catalog=None,
        config=None,
        logger=None,
        images=None,
        paidiverpy=None,
        step_name=None,
        parameters=None,
        config_index=None,
        raise_error=False,
        verbose=True,
        n_jobs=1,
    ):

        super().__init__(
            config_file_path=config_file_path,
            input_path=input_path,
            output_path=output_path,
            catalog_path=catalog_path,
            catalog_type=catalog_type,
            catalog=catalog,
            config=config,
            logger=logger,
            images=images,
            paidiverpy=paidiverpy,
            raise_error=raise_error,
            verbose=verbose,
            n_jobs=n_jobs,
        )

        self.step_name = step_name
        if parameters:
            self.config_index = self.config.add_step(config_index, parameters)
        self.step_metadata = self._calculate_steps_metadata(
            self.config.steps[self.config_index]
        )

    def run(self, add_new_step=True):
        mode = self.step_metadata.get("mode")
        if not mode:
            raise ValueError("The mode is not defined in the configuration file.")
        test = self.step_metadata.get("test")
        params = self.step_metadata.get("params") or {}
        method, params = self._get_method_by_mode(params, CONVERT_LAYER_METHODS, mode)
        images = self.images.get_step(step=len(self.images.images) - 1, by_order=True)
        if self.n_jobs == 1:
            image_list = self.process_sequentially(images, method, params)
        else:
            image_list = self.process_parallel(images, method, params)
        if not test:
            self.step_name = (
                f"convert_{self.config_index}" if not self.step_name else self.step_name
            )
            if add_new_step:
                self.images.add_step(
                    step=self.step_name,
                    images=image_list,
                    step_metadata=self.step_metadata,
                    catalog=self.get_catalog(),
                )
            else:
                self.images.images[-1] = image_list
                return self.images

    def process_sequentially(self, images, method, params):
        image_list = [method(img, params=params) for img in images]
        return image_list

        # image_list = []
        # for img in tqdm(images, total=len(images), desc="Processing Images"):
        #     img = method(img, params=params)
        #     image_list.append(img)
        # return image_list

        # mask = images.mask
        # data = images.data
        # data = np.array([method(img, params=params) for img in data])
        # if len(data.shape) == 3:
        #     data = np.expand_dims(data, axis=-1)
        # if data.shape[-1] == 1 and mask.shape[-1] == 3:
        #     mask = np.expand_dims(mask[:, :,:, 0], axis=-1)
        # image_list = np.ma.MaskedArray(data, mask=mask)
        # return image_list

    def process_parallel(self, images, method, params):
        delayed_images = [dask.delayed(method)(img, params) for img in images]
        with dask.config.set(scheduler='threads', num_workers=self.n_jobs):
            self.logger.info("Processing images using %s cores", self.n_jobs)
            with ProgressBar():
                delayed_images = compute(*delayed_images)
        image_list = [da.from_array(img) for img in delayed_images]
        return image_list

        # processed_chunks = processed_chunks[0]
        # if len(processed_chunks.shape) == 3:
        #     processed_chunks = np.expand_dims(processed_chunks, axis=-1)
        # processed_data = da.from_array(processed_chunks, chunks=(1, processed_chunks.shape[1], processed_chunks.shape[2], processed_chunks.shape[3])).squeeze()

        # if len(processed_data.shape) == 3:
        #     processed_data = np.expand_dims(processed_data, axis=-1)
        # if processed_data.shape[-1] == 1 and mask.shape[-1] == 3:
        #     mask = np.expand_dims(mask[:, :, :, 0], axis=-1)

        # image_list = da.ma.masked_array(processed_data, mask=mask)


        # mask = da.ma.getmaskarray(images)
        # data = dask.array.ma.getdata(images)

        # @delayed
        # def apply_function_wrapper(chunk: np.ndarray, params) -> np.ndarray:
        #     return np.array([method(image, params) for image in chunk])
        # delayed_image_chunks = apply_function_wrapper(data, params)
        # with dask.config.set(scheduler='threads', num_workers=self.n_jobs):
        #     self.logger.info("Processing images using %s cores", self.n_jobs)
        #     with ProgressBar():
        #         processed_chunks = compute(delayed_image_chunks)

        # processed_chunks = processed_chunks[0]
        # if len(processed_chunks.shape) == 3:
        #     processed_chunks = np.expand_dims(processed_chunks, axis=-1)
        # processed_data = da.from_array(processed_chunks, chunks=(1, processed_chunks.shape[1], processed_chunks.shape[2], processed_chunks.shape[3])).squeeze()

        # if len(processed_data.shape) == 3:
        #     processed_data = np.expand_dims(processed_data, axis=-1)
        # if processed_data.shape[-1] == 1 and mask.shape[-1] == 3:
        #     mask = np.expand_dims(mask[:, :, :, 0], axis=-1)

        # image_list = da.ma.masked_array(processed_data, mask=mask)
        # return image_list

    def convert_bits(self, image_data, params: BitParams = BitParams()):
        if params.output_bits == 8:
            image_data = np.uint8(image_data * 255)
        elif params.output_bits == 16:
            image_data = np.uint16(image_data * 65535)
        elif params.output_bits == 32:
            image_data = np.float32(image_data)
        else:
            self.logger.warning("Unsupported output bits: %s", params.output_bits)
            if self.raise_error:
                raise ValueError(f"Unsupported output bits: {params.output_bits}")

        return image_data

    def channel_convert(self, image_data, params: ToParams = ToParams()):
        try:
            if params.to == "RGB":
                if image_data.shape[-1] == 1:
                    image_data = cv2.cvtColor(image_data, cv2.COLOR_GRAY2RGB)
                else:
                    raise ValueError("The image is already in RGB format.")
            elif params.to == "gray":
                if image_data.shape[-1] == 1:
                    raise ValueError("The image is already in grayscale.")
                if params.channel_selector in [0, 1, 2]:
                    image_data = image_data[:, :, params.channel_selector]
                else:
                    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            self.logger.error(f"Failed to convert the image to {params.to}: {str(e)}")
            if self.raise_error:
                raise ValueError(
                    f"Failed to convert the image to {params.to}: {str(e)}"
                ) from e
        return image_data

    def get_bayer_pattern(self, image_data, params: BayerPatternParams = BayerPatternParams()):
        # Determine the number of channels in the input image
        if image_data.shape[-1] != 1:
            self.logger.warning(
                "Invalid Bayer pattern for a single-channel image: %s",
                params.bayer_pattern,
            )
            if self.raise_error:
                raise ValueError(
                    "Invalid Bayer pattern for a single-channel image. Expected 'RG', 'BG', 'GR', or 'GB'."
                )
            return image_data
        try:
            bayer_pattern = {
                "RG": cv2.COLOR_BAYER_RG2RGB,
                "BG": cv2.COLOR_BAYER_BG2RGB,
                "GR": cv2.COLOR_BAYER_GR2RGB,
                "GB": cv2.COLOR_BAYER_GB2RGB
            }[params.bayer_pattern]
        except KeyError as exc:
            self.logger.warning(
                "Invalid Bayer pattern for a single-channel image: %s",
                params.bayer_pattern,
            )
            if self.raise_error:
                raise exc

            return image_data
        image_data = cv2.cvtColor(image_data, bayer_pattern)
        
        return image_data

    def normalize_image(self, image_data, params: NormalizeParams = NormalizeParams()):
        try:
            return cv2.normalize(
                image_data, image_data, params.min, params.max, cv2.NORM_MINMAX, dtype=cv2.CV_32F
            )
        except Exception as e:
            self.logger.error(f"Failed to normalize the image: {str(e)}")
            if self.raise_error:
                raise ValueError(f"Failed to normalize the image: {str(e)}") from e
        return image_data

    # if params.autoscale:
        #     try:
        #         min_val = np.min(image_data, axis=(0, 1), keepdims=True)
        #         max_val = np.max(image_data, axis=(0, 1), keepdims=True)
        #         image_data = (image_data - min_val) / (max_val - min_val)
        #     except Exception as e:
        #         self.logger.error("Failed to autoscale the image: %s", e)
        #         if self.raise_error:
        #             raise ValueError(f"Failed to autoscale the image: {str(e)}") from e
        #     return image_data

    def resize(self, image_data, params: ResizeParams = ResizeParams()):
        try:
            return cv2.resize(
                image_data, (params.min, params.max), interpolation=cv2.INTER_LANCZOS4
            )
        except Exception as e:
            self.logger.error(f"Failed to resize the image: {str(e)}")
            if self.raise_error:
                raise ValueError(f"Failed to resize the image: {str(e)}") from e
        return image_data

    def crop_images(self, image_data, params: CropParams = CropParams()):
        try:
            start_x, end_x = params.x[0]
            start_y, end_y = params.y[1]
            if (
                start_x < 0
                or end_x > image_data.shape[1]
                or start_y < 0
                or end_y > image_data.shape[2]
            ):
                raise ValueError("Crop range is out of bounds.")
            return image_data[:, start_x:end_x, start_y:end_y, :]
        except Exception as e:
            self.logger.error("Failed to crop the image: %s", str(e))
            if self.raise_error:
                raise ValueError(f"Failed to crop the image: {str(e)}") from e
        return image_data
