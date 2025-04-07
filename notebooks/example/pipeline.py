# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: torch
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Inference Pipeline

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import queue
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import mlflow.pyfunc
import numpy as np
import torch
from common import ONNXModel, normalize, preprocess_depth_anything_v2, sigmoid
from PIL import Image
from tqdm.auto import tqdm
from transformers import Pipeline, pipeline

# %% [markdown]
# ## Load Models
#
# This expects that the models are already downloaded and available in the `models` directory. You can download the models with [`download_models.py`](download_models.py).

# %%
mde_model_source: Literal['huggingface', 'onnx'] = 'onnx'
seg_model_source: Literal['mlflow', 'onnx'] = 'onnx'
ae_model_source: Literal['mlflow'] = 'mlflow'

# %%
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

# Load segmentation model
if seg_model_source == 'onnx':
    seg_model = ONNXModel('models/seg/model/model.onnx')
    seg_input_shape = seg_model.input_shape
elif seg_model_source == 'mlflow':
    seg_model = mlflow.pyfunc.load_model(
        'models/seg/model', model_config={'device': DEVICE}
    )
    seg_input_schema = seg_model.metadata.get_input_schema()
    assert seg_input_schema is not None, (
        'Input schema is not available for the segmentation model'
    )
    seg_input_shape = seg_input_schema.inputs[0].shape
else:
    raise ValueError(f'Unsupported segmentation model source: {seg_model_source}')

# Load autoencoder model
if ae_model_source == 'mlflow':
    ae_model = mlflow.pyfunc.load_model(
        'models/ae/model', model_config={'device': DEVICE}
    )
    ae_input_schema = ae_model.metadata.get_input_schema()
    assert ae_input_schema is not None, (
        'Input schema is not available for the autoencoder model'
    )
    ae_input_shape = ae_input_schema.inputs[0].shape
else:
    raise ValueError(f'Unsupported autoencoder model source: {ae_model_source}')

# Load depth estimation model
if mde_model_source == 'onnx':
    mde_model = ONNXModel('models/depth_anything_v2_vits_dynamic.onnx')
elif mde_model_source == 'huggingface':
    mde_model = pipeline(
        task='depth-estimation',
        model='depth-anything/Depth-Anything-V2-Small-hf',
        device=DEVICE,
    )
else:
    raise ValueError(f'Unsupported depth model source: {mde_model_source}')

print(f'Autoencoder model input shape: (B, T, C, H, W) = {ae_input_shape}')
print(f'Segmentation model input shape: (B, C, H, W) = {seg_input_shape}')

# Get temporal dimension from autoencoder model
T = ae_input_shape[1]


# %% [markdown]
# ## Segmentation and MDE Inference Helper Functions


# %%
def get_mde(model: Pipeline, img: np.ndarray) -> tuple[np.ndarray, float]:
    """Get depth estimation from the model."""
    start_time = time.time()
    img = cv2.resize(img, (518, 518))  # default size for Depth-Anything
    img_pil = Image.fromarray(img)  # type: ignore
    depth = model(img_pil)['depth']  # type: ignore
    depth = np.array(depth).astype(np.float32) / 255.0
    processing_time = time.time() - start_time
    return depth, processing_time


def get_mde_maps(
    model: Pipeline, images: list[np.ndarray]
) -> tuple[list[np.ndarray], float]:
    """Get batched depth estimation from the model."""
    start_time = time.time()
    images = [cv2.resize(img, (518, 518)) for img in images]
    images_pil = [Image.fromarray(img) for img in images]  # type: ignore
    depths = model(images_pil)
    depths = [
        normalize(depth['predicted_depth'].numpy())  # type: ignore
        for depth in depths  # type: ignore
    ]
    processing_time = time.time() - start_time
    return depths, processing_time / len(images)


def get_mde_onnx(
    model: ONNXModel, images: list[np.ndarray]
) -> tuple[list[np.ndarray], float]:
    """Get batched depth estimation from the ONNX model."""
    start_time = time.time()
    images = [
        preprocess_depth_anything_v2(cv2.resize(img, (518, 518))) for img in images
    ]
    batch = np.concatenate(images, axis=0)
    depth_maps = model.predict(batch)
    depth_maps = [normalize(depth_map) for depth_map in depth_maps]
    processing_time = time.time() - start_time
    return depth_maps, processing_time / len(images)


def get_mask(
    model: mlflow.pyfunc.PyFuncModel, img: np.ndarray, img_size: tuple[int, int]
) -> tuple[np.ndarray, float]:
    """Get mask from the model."""
    start_time = time.time()
    img = cv2.resize(img, img_size)
    img = np.transpose(img, (2, 0, 1))  # HWC -> CWH
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    mask = torch.Tensor(prediction).sigmoid()
    mask_numpy = mask.squeeze().numpy()
    binary_mask = (mask_numpy > 0.5).astype(np.uint8)
    processing_time = time.time() - start_time
    return binary_mask, processing_time


def get_masks(
    model: mlflow.pyfunc.PyFuncModel,
    images: list[np.ndarray],
    img_size: tuple[int, int],
) -> tuple[list[np.ndarray], float]:
    """Get batched masks from the model."""
    start_time = time.time()
    images = [
        np.transpose(cv2.resize(img, img_size), (2, 0, 1)).astype(np.float32)
        for img in images
    ]
    batch = np.stack(images, axis=0)
    predictions = model.predict(batch)
    binary_masks = [(sigmoid(p.squeeze()) > 0.5).astype(np.uint8) for p in predictions]
    processing_time = time.time() - start_time
    return binary_masks, processing_time / len(images)


def get_masks_onnx(
    model: ONNXModel, images: list[np.ndarray], img_size: tuple[int, int]
) -> tuple[list[np.ndarray], float]:
    """Get batched masks from the ONNX model."""
    start_time = time.time()
    images = [cv2.resize(img, img_size).astype(np.float32) for img in images]
    batch = np.stack(images, axis=0)
    batch = np.transpose(batch, (0, 3, 1, 2))  # HWC -> NCHW
    masks = model.predict(batch)
    masks = [sigmoid(mask.squeeze()) for mask in masks]
    binary_masks = [(mask > 0.5).astype(np.uint8) for mask in masks]
    processing_time = time.time() - start_time
    return binary_masks, processing_time / len(images)


def mask_mde(
    mde: np.ndarray, mask: np.ndarray, output_shape: tuple[int, int]
) -> tuple[np.ndarray, float]:
    """Apply mask to the depth estimation."""
    start_time = time.time()
    mask = cv2.resize(mask, output_shape)
    mde = cv2.resize(mde, output_shape)
    masked = mde * mask
    processing_time = time.time() - start_time
    return masked, processing_time


# %% [markdown]
# ## Parallel Inference


# %%
class ImageLoader(threading.Thread):
    """Thread that loads images and puts them in the queue."""

    def __init__(self, image_paths: list[Path], output_queue: queue.Queue) -> None:
        threading.Thread.__init__(self)
        self.image_paths = image_paths
        self.output_queue = output_queue
        self.daemon = True

    def run(self) -> None:
        """Load images and put them in the output queue."""
        for path in self.image_paths:
            start_time = time.time()
            img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            loading_time = time.time() - start_time
            self.output_queue.put((img, {'load_time': loading_time}))
        # Signal end of data with None
        self.output_queue.put(None)


class ParallelProcessor(threading.Thread):
    """Thread that runs MDE and SEG models in parallel."""

    def __init__(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        mde_model: Pipeline | ONNXModel,
        seg_model: mlflow.pyfunc.PyFuncModel | ONNXModel,
        batch_size: int = 2,
    ) -> None:
        threading.Thread.__init__(self)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.mde_model = mde_model
        self.seg_model = seg_model
        self.batch_size = batch_size
        self.seg_input_shape_ = (
            seg_model.metadata.get_input_schema().inputs[0].shape[-2:]  # type: ignore
            if isinstance(seg_model, mlflow.pyfunc.PyFuncModel)
            else seg_model.input_shape[-2:]
        )
        self.mde_func_: (
            Callable[[ONNXModel, list[np.ndarray]], tuple[list[np.ndarray], float]]
            | Callable[[Pipeline, list[np.ndarray]], tuple[list[np.ndarray], float]]
        ) = get_mde_onnx if isinstance(mde_model, ONNXModel) else get_mde_maps
        self.mask_func_: (
            Callable[
                [ONNXModel, list[np.ndarray], tuple[int, int]],
                tuple[list[np.ndarray], float],
            ]
            | Callable[
                [mlflow.pyfunc.PyFuncModel, list[np.ndarray], tuple[int, int]],
                tuple[list[np.ndarray], float],
            ]
        ) = get_masks_onnx if isinstance(seg_model, ONNXModel) else get_masks
        self.daemon = True

    def run(self) -> None:
        """Process images with MDE and SEG models in parallel using batch processing."""
        batch_images = []
        batch_times = []

        while True:
            # Try to fill a batch
            while len(batch_images) < self.batch_size:
                if self.input_queue.empty():
                    break

                item = self.input_queue.get()

                # Check for end of data
                if item is None:
                    self.input_queue.task_done()

                    # Process any remaining images in the batch
                    if batch_images:
                        self._process_batch(batch_images, batch_times)

                    self.output_queue.put(None)
                    return

                img, timing_info = item
                batch_images.append(img)
                batch_times.append(timing_info)

            # Process the batch
            if batch_images:
                self._process_batch(batch_images, batch_times)
                batch_images = []
                batch_times = []

    def _process_batch(
        self, batch_images: list[np.ndarray], batch_times: list[dict[str, float]]
    ) -> None:
        """Process a batch of images with MDE and SEG models."""
        # Process MDE and mask in parallel using threads
        mde_results = [None, 0]
        mask_results = [None, 0]

        def process_mde_batch() -> None:
            mde_results[0], mde_results[1] = self.mde_func_(
                self.mde_model, batch_images
            )

        def process_mask_batch() -> None:
            mask_results[0], mask_results[1] = self.mask_func_(
                self.seg_model, batch_images, self.seg_input_shape_[-2:]
            )

        start_time = time.time()
        # TODO: use a pool of threads instead of creating new threads each time
        mde_thread = threading.Thread(target=process_mde_batch)
        mask_thread = threading.Thread(target=process_mask_batch)

        mde_thread.start()
        mask_thread.start()

        mde_thread.join()
        mask_thread.join()
        parallel_time = time.time() - start_time

        mde_batch = mde_results[0]
        mask_batch = mask_results[0]

        # Process each item in the batch and put results in output queue
        for i, (img, timing_info) in enumerate(zip(batch_images, batch_times)):
            # Get individual results from batch
            mde = mde_batch[i] if isinstance(mde_batch, list) else mde_batch
            mask = mask_batch[i] if isinstance(mask_batch, list) else mask_batch

            # Combine results
            masked_mde, masking_time = mask_mde(mde, mask, (64, 64))

            # Update timing info
            timing_info.update(
                {
                    # Average time per image
                    'mde_time': mde_results[1],
                    'mask_time': mask_results[1],
                    'masking_time': masking_time,
                    'parallel_time': parallel_time / self.batch_size,
                }
            )

            # Put results in output queue
            self.output_queue.put((img, mde, mask, masked_mde, timing_info))
            self.input_queue.task_done()


class AutoencoderProcessor(threading.Thread):
    """Thread that processes frames with the autoencoder model."""

    def __init__(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        ae_model: mlflow.pyfunc.PyFuncModel,
        temporal_dimension: int,
    ) -> None:
        threading.Thread.__init__(self)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.ae_model = ae_model
        if temporal_dimension > 0:
            self.temporal_dimension = temporal_dimension
        else:
            raise ValueError('Temporal dimension must be greater than 0')
        self.buffer = deque(maxlen=temporal_dimension)
        self.img_buffer = deque(maxlen=temporal_dimension)
        self.mde_buffer = deque(maxlen=temporal_dimension)
        self.mask_buffer = deque(maxlen=temporal_dimension)
        self.timing_buffer = deque(maxlen=temporal_dimension)
        self.daemon = True

    def run(self) -> None:
        """Process frames with the autoencoder model."""
        waiting_for_frames = True

        while waiting_for_frames:
            item = self.input_queue.get()

            # Check for end of data
            if item is None:
                waiting_for_frames = False
                self.input_queue.task_done()
                continue

            img, mde, mask, masked_mde, timing_info = item
            self.buffer.append(masked_mde)
            self.img_buffer.append(img)
            self.mde_buffer.append(mde)
            self.mask_buffer.append(mask)
            self.timing_buffer.append(timing_info)

            # Process when we have enough frames or reached the end
            if len(self.buffer) == self.temporal_dimension or not waiting_for_frames:
                while len(self.buffer) < self.temporal_dimension:
                    # Duplicate the last frame to fill the buffer
                    self.buffer.append(self.buffer[-1])

                # Create stacked input for autoencoder
                stacked = np.stack(list(self.buffer), axis=0)
                stacked = np.expand_dims(stacked, axis=0)  # Add batch dimension
                stacked = np.expand_dims(stacked, axis=2)  # Add channel dimension

                # Process with autoencoder
                start_time = time.time()
                reconstructed = self.ae_model.predict(stacked)
                reconstructed = reconstructed.squeeze()
                ae_time = time.time() - start_time

                # Calculate differences between masked_mde and reconstruction
                differences = []
                for i in range(min(len(self.buffer), len(reconstructed))):
                    diff = np.abs(self.buffer[i] - reconstructed[i])
                    differences.append(diff)

                # Update timing info
                for i in range(self.temporal_dimension):
                    self.timing_buffer[i].update(
                        {
                            # Average time per image
                            'ae_time': ae_time / self.temporal_dimension,
                        }
                    )

                # Output result for each frame in the temporal window
                for i in range(min(len(self.buffer), len(reconstructed))):
                    self.output_queue.put(
                        (
                            self.img_buffer[i],
                            self.mask_buffer[i],
                            self.mde_buffer[i],
                            self.buffer[i],
                            reconstructed[i] if i < len(reconstructed) else None,
                            differences[i] if i < len(differences) else None,
                            self.timing_buffer[i],
                        )
                    )

                # Clear buffer, by sliding by T frames at a time.
                # (Alternatively, could be 1 for sliding window approach
                #  with overlap, but this is more efficient.)
                for _ in range(self.temporal_dimension):
                    self.buffer.popleft()
                    self.img_buffer.popleft()
                    self.mde_buffer.popleft()
                    self.mask_buffer.popleft()
                    self.timing_buffer.popleft()

            self.input_queue.task_done()

        # Signal end of data
        self.output_queue.put(None)


# %% [markdown]
# ## Visualization


# %%
def create_mosaic(
    original: np.ndarray,
    mask: np.ndarray,
    mde: np.ndarray,
    masked_mde: np.ndarray,
    reconstruction: np.ndarray | None,
    difference: np.ndarray | None,
    timing_info: dict[str, float],
    output_size: tuple[int, int] = (1080, 720),
) -> np.ndarray:
    """Create a 3x2 mosaic of the different processing stages with timing information."""
    # Resize all inputs to a standard size for display
    h, w = output_size[1] // 2, output_size[0] // 3

    # Create a blank canvas
    mosaic = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)

    # Helper function to resize and add timing text
    def process_image(
        img: np.ndarray | None,
        title: str,
        timing_key: str | None = None,
        cmap: str | None = None,
    ) -> np.ndarray:
        if img is None:
            blank = np.zeros((h, w, 3), dtype=np.uint8)
            blank[:] = (50, 50, 50)  # Dark gray
            return blank

        # Convert to 3-channel if grayscale
        if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
            if cmap:
                # Apply colormap for visualization
                normalized = img / np.max(img) if np.max(img) > 0 else img
                colored = getattr(cm, cmap)(normalized)
                img_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
            else:
                # Standard grayscale to RGB conversion
                img_rgb = np.stack([img, img, img], axis=2)
                if len(img_rgb.shape) == 3 and img_rgb.shape[2] == 3:
                    img_rgb = (
                        (img_rgb * 255).astype(np.uint8)
                        if img_rgb.max() <= 1.0
                        else img_rgb.astype(np.uint8)
                    )
        else:
            img_rgb = img

        # Make sure we have a valid RGB image before resizing
        if img_rgb.dtype != np.uint8:
            img_rgb = (
                (img_rgb * 255).astype(np.uint8)
                if img_rgb.max() <= 1.0
                else img_rgb.astype(np.uint8)
            )

        # Resize to exact dimensions needed for the mosaic cell
        img_resized = cv2.resize(img_rgb, (w, h))

        # Add timing information if available
        if timing_key and timing_key in timing_info:
            time_text = f'{title}: {1000 * timing_info[timing_key]:.1f}ms'
        else:
            time_text = title

        # Add text to the image
        img_with_text = img_resized.copy()
        cv2.putText(
            img_with_text,
            time_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return img_with_text

    # Process each image and add to mosaic
    original_processed = process_image(original, 'Original', 'load_time')
    mask_processed = process_image(mask, 'Mask', 'mask_time')
    mde_processed = process_image(mde, 'MDE', 'mde_time', cmap='inferno')
    masked_mde_processed = process_image(
        masked_mde, 'Masked MDE', 'masking_time', cmap='inferno'
    )

    if reconstruction is not None:
        recon_processed = process_image(
            reconstruction, 'Reconstruction', 'ae_time', cmap='inferno'
        )
    else:
        recon_processed = process_image(None, 'Reconstruction')

    if difference is not None:
        diff_processed = process_image(difference, 'Difference', None, cmap='hot')
    else:
        diff_processed = process_image(None, 'Difference')

    def assert_shape(
        arr: np.ndarray, expected_shape: tuple[int, int, int], name: str = ''
    ) -> None:
        """Check if the array has the expected shape."""
        assert arr.shape == expected_shape, (
            f'Shape mismatch for {name}: {arr.shape} != {expected_shape}'
        )

    # Perform explicit checks on dimensions to ensure they match
    assert_shape(original_processed, (h, w, 3), 'Original')
    assert_shape(mask_processed, (h, w, 3), 'Mask')
    assert_shape(mde_processed, (h, w, 3), 'MDE')
    assert_shape(masked_mde_processed, (h, w, 3), 'Masked MDE')
    assert_shape(recon_processed, (h, w, 3), 'Reconstruction')
    assert_shape(diff_processed, (h, w, 3), 'Difference')

    # Arrange in 3x2 grid
    # Top row
    mosaic[0:h, 0:w] = original_processed
    mosaic[0:h, w : 2 * w] = mask_processed
    mosaic[0:h, 2 * w : 3 * w] = mde_processed

    # Bottom row
    mosaic[h : 2 * h, 0:w] = masked_mde_processed
    mosaic[h : 2 * h, w : 2 * w] = recon_processed
    mosaic[h : 2 * h, 2 * w : 3 * w] = diff_processed

    # Add overall timing information
    total_time = timing_info['total_time']
    cv2.putText(
        mosaic,
        f'Total processing time: {1000 * total_time:.1f}ms, FPS: {1 / total_time:.0f}',
        (10, output_size[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return mosaic


# %% [markdown]
# ## Main pipeline execution with video output
#


# %%
def run_pipeline_with_visualization(
    image_paths: list[Path], output_video_path: str, fps: float = 24.0
) -> dict[str, list[float]]:
    # Create queues
    image_queue = queue.Queue(maxsize=8)
    processed_queue = queue.Queue(maxsize=8)
    result_queue = queue.Queue()

    # Start threads
    image_loader = ImageLoader(image_paths, image_queue)
    parallel_processor = ParallelProcessor(
        image_queue,
        processed_queue,
        mde_model,
        seg_model,
    )
    ae_processor = AutoencoderProcessor(processed_queue, result_queue, ae_model, T)

    image_loader.start()
    parallel_processor.start()
    ae_processor.start()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = None  # Will be initialized with the first frame
    performance_stats = defaultdict(list)

    with tqdm(total=len(image_paths), desc='Processing frames') as pbar:
        while True:
            result = result_queue.get()
            if result is None:
                break

            (
                img,
                mask,
                mde,
                masked_mde,
                reconstruction,
                difference,
                timing_info,
            ) = result

            # Calculate total processing time
            total_time = sum(
                [
                    timing_info.get('load_time', 0),
                    timing_info.get('parallel_time', 0),
                    timing_info.get('masking_time', 0),
                    timing_info.get('ae_time', 0),
                ]
            )
            timing_info['total_time'] = total_time

            # Update performance statistics
            for key, value in timing_info.items():
                performance_stats[key].append(value)

            # Create mosaic frame
            # TODO: average the times over N frames
            mosaic = create_mosaic(
                img,
                mask,
                mde,
                masked_mde,
                reconstruction,
                difference,
                timing_info,
            )

            # Initialize video writer with the first frame
            if video_writer is None:
                video_writer = cv2.VideoWriter(
                    output_video_path, fourcc, fps, (mosaic.shape[1], mosaic.shape[0])
                )

            # Write frame to video
            video_writer.write(cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR))

            result_queue.task_done()
            pbar.update(1)

    # Release video writer
    if video_writer is not None:
        video_writer.release()

    # Wait for all threads to complete
    image_loader.join()
    parallel_processor.join()
    ae_processor.join()

    return performance_stats


# %%
# Execute the pipeline and generate the mosaic video
image_paths = sorted(Path('images').glob('*.jpg'), key=lambda x: int(x.stem))
output_video_path = 'pipeline_visualization_30fps.mp4'

performance_stats = run_pipeline_with_visualization(
    image_paths, output_video_path, fps=30.0
)

# %%
for k, v in performance_stats.items():
    print(f'{k}: {np.mean(v):.3f} ± {np.std(v):.3f}')

# %%
# Display performance summary
performance_summary = {}
for key, values in performance_stats.items():
    if values:
        performance_summary[key] = {
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'total': sum(values),
        }

print('Performance Summary (in seconds):')
print('-' * 50)
for key, stats in performance_summary.items():
    print(f'{key.replace("_", " ").title()}:')
    print(f'  Min: {stats["min"]:.3f}s')
    print(f'  Max: {stats["max"]:.3f}s')
    print(f'  Avg: {stats["avg"]:.3f}s')
    print(f'  Total: {stats["total"]:.3f}s')
    print('-' * 50)

print(f'Video saved to: {output_video_path}')

# %%
# Generate performance visualization
keys = list(performance_summary.keys())
avg_times = [performance_summary[k]['avg'] for k in keys]

plt.figure(figsize=(12, 6))
bars = plt.bar(
    keys,
    avg_times,
)

plt.title('Average Processing Time per Stage')
plt.xlabel('Processing Stage')
plt.ylabel('Time (seconds)')
plt.xticks(rotation=45)

# Add the values on top of the bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f'{1000 * height:.1f}ms',
        ha='center',
        va='bottom',
    )

plt.tight_layout()

# add FPS to legend
fps = 1 / np.mean(performance_stats['total_time'])
plt.legend([f'FPS: {fps:.2f}'], loc='upper left')
plt.savefig('performance_chart.png')
plt.show()
