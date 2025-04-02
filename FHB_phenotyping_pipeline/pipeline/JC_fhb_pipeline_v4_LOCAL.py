# # Create fhb_env
# conda create --name fhb_env python=3.9

# # Activate fhb_env
# conda activate fhb_env

# # Install libraries
# pip install numpy
# pip install tensorflow
# pip install opencv-python # Instad of cv2
# pip install pillow # Instead of PIL
# pip install pandas
# pip install scipy
# pip install matplotlib
# pip install plotly


# Install librareis round 2
# pip install tensorflow-object-detection-api
# pip install protobuf==3.20.3


#### TO RUN ####
# Use sudo python to overcome protoc permissions

# Libraries

import numpy as np
import tensorflow as tf
import cv2
import os
import io
import hashlib
import google.protobuf
from typing import Any, Callable, Dict, List, Optional, Text, Tuple, Sequence
import sys # To access input and output. Built in to python, don't need to install.
import re # To create regex. Built in to python, don't need to install.

import subprocess
from io import BytesIO
from PIL import Image
import google.protobuf
import google.protobuf.text_format as text_format
from google.protobuf.any_pb2 import Any
from google.protobuf.timestamp_pb2 import Timestamp
from google.protobuf.descriptor_pb2 import FileDescriptorSet
from object_detection.core import standard_fields as fields
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import np_box_list
from object_detection.utils import np_box_list_ops

from ast import literal_eval
import pandas as pd
from PIL import Image
from scipy.ndimage import rotate

# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px

from tensorflow.python.lib.io import file_io

### ADDED 7/2 for 4-Row Plots ###
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

############################################################################################################

# Constants related to object detection results
# Constant representing the detection scores output by the object detection model
_DETECTION_SCORES = fields.DetectionResultFields.detection_scores
# Constant representing the detection boxes output by the object detection model
_DETECTION_BOXES = fields.DetectionResultFields.detection_boxes
# Constant representing the detected classes output by the object detection model
_DETECTION_CLASSES = fields.DetectionResultFields.detection_classes
# Maximum number of boxes to consider in non-maximum suppression (NMS)
_MAX_NUM_BOXES_NMS = 100000
# List of valid output keys expected from the object detection model
_VALID_OUTPUT_KEYS = ['SemanticProbabilities:0', 'output', 'output_layer']

def _bytes_feature(value: tf.Tensor) -> tf.train.Feature:
  """Returns a bytes_list from a string / byte."""
  if tf.executing_eagerly():
    # BytesList won't unpack a string from an EagerTensor.
    value = value.numpy()
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _np_img_to_serialized_tf_example(img: np.ndarray) -> bytes:
  """Creates a serialized Tensorflow Example from a Numpy image array."""
  # Optionally convert to non-normalized array before jpeg conversion.
  convert = img.max() <= 1.0
  jpeg_encoded_img_feature = _bytes_feature(
      tf.io.encode_jpeg(img * 255.0 if convert else img))
  return tf.train.Example(
      features=tf.train.Features(
          feature={'image/encoded': jpeg_encoded_img_feature
                  })).SerializeToString()

_DETECTION_SCORES = fields.DetectionResultFields.detection_scores
_DETECTION_BOXES = fields.DetectionResultFields.detection_boxes
_DETECTION_CLASSES = fields.DetectionResultFields.detection_classes

def _convert_boxes_from_normalized_to_pixel_space(
    img: np.ndarray, boxes: np.ndarray) -> np.ndarray:
  """Converts normalized box coordinates to coordinates of the img."""
  img_h, img_w, _ = img.shape
  normalized_boxes = boxes.copy()
  normalized_boxes[:, [0, 2]] *= img_h
  # boxes[:, 0] *= img_h
  # boxes[:, 2] *= img_h
  # boxes[:, 1] *= img_w
  # boxes[:, 3] *= img_w
  normalized_boxes[:, [1, 3]] *= img_w
  return np.around(normalized_boxes)

def _nms_overall(
      boxes_array: np.ndarray,
      classes_array: np.ndarray,
      scores_array: np.ndarray,
      min_confidence: float,
      nms_overlapping_threshold: float,
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Performs non-max suppression for all classes altogether.

    Args:
      boxes_array: [N, 4] float32 array of locations following this format:
        [box_top, box_left, box_bottom, box_right]
      classes_array: [N] float32 array of class labels for top predicted class.
      scores_array: [N] float32 array of model confidence scores.
      min_confidence: Filters all predictions with confidence scores below this
        value.
      nms_overlapping_threshold: Overlap between two boxes required before
        performing non-max suppression.

    Returns:
      The tuple of the boxes after NMS, their classes, and scores in the form
      (boxes_after_nms, classes_after_nms, scores_after_nms). The data structure
      and types are the same as corresponding inputs.
    """
    pre_nms_boxlist = np_box_list.BoxList(boxes_array)
    pre_nms_boxlist.add_field('classes', classes_array)
    pre_nms_boxlist.add_field('scores', scores_array)
    nms_boxlist = np_box_list_ops.non_max_suppression(
        pre_nms_boxlist,
        max_output_size=_MAX_NUM_BOXES_NMS,
        score_threshold=min_confidence,
        iou_threshold=nms_overlapping_threshold
    )
    return (
        nms_boxlist.get(),
        nms_boxlist.get_field('classes'),
        nms_boxlist.get_field('scores')
    )

def _nms_per_class(
      boxes_array: np.ndarray,
      classes_array: np.ndarray,
      scores_array: np.ndarray,
      min_confidence: float,
      nms_overlapping_threshold: float,
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Performs non-max suppression on a per-class basis.

    Args:
      boxes_array: [N, 4] float32 array of locations following this format:
        [box_top, box_left, box_bottom, box_right]
      classes_array: [N] float32 array of class labels for top predicted class.
      scores_array: [N] float32 array of model confidence scores.
      min_confidence: Filters all predictions with confidence scores below this
        value.
      nms_overlapping_threshold: Overlap between two boxes required before
        performing non-max suppression.

    Returns:
      The tuple of the boxes after NMS, their classes, and scores in the form
      (boxes_after_nms, classes_after_nms, scores_after_nms). The data structure
      and types are the same as corresponding inputs.
    """
    class_nms_boxes_list = []
    class_nms_classes_list = []
    class_nms_scores_list = []
    for class_label in np.unique(classes_array):
      class_indices = np.where(classes_array == class_label)
      class_nms_boxes, class_nms_classes, class_nms_scores = _nms_overall(
          boxes_array=boxes_array[class_indices],
          classes_array=classes_array[class_indices],
          scores_array=scores_array[class_indices],
          min_confidence=min_confidence,
          nms_overlapping_threshold=nms_overlapping_threshold,
      )
      class_nms_boxes_list.append(class_nms_boxes)
      class_nms_classes_list.append(class_nms_classes)
      class_nms_scores_list.append(class_nms_scores)

    return (
        np.concatenate(class_nms_boxes_list),
        np.concatenate(class_nms_classes_list),
        np.concatenate(class_nms_scores_list)
    )

def post_process_detections(boxes, classes, scores, min_confidence, nms_threshold):
    # Implement post-processing steps here based on your Centernet model output
    # You may need to extract bounding boxes, confidence scores, etc., and apply NMS
    # This function should return the filtered detections
    # Adjust this based on the actual structure of your Centernet model output
    nms_boxes, nms_classes, nms_scores = _nms_per_class(
        boxes, classes, scores, min_confidence, nms_threshold)

    return {
        'boxes': nms_boxes,
        'classes': nms_classes,
        'scores': nms_scores,
    }

def resize_img(imgs: np.ndarray, width: int, height: int,
               interpolation: int = cv2.INTER_AREA) -> np.ndarray:
  """Resizes input image to new specified shape.

  The imgs arg can be either a single image or a batch of images.
  Supported shapes are:
    - single-shannel image: [H, W]
    - multi-channel image: [H, W, C]
    - multi-channel image batch: [B, H, W, C]

  Args:
    imgs: array representing input image(s)
    width: resize image width to this value
    height: resize image height to this value
    interpolation: resize interpolation method

  Returns:
    Array with resized image(s). Please note that this function will preseve the
    number of dimensions of the original input, and will only change the size of
    the image in the height and width dimensions. E.g. if the input is an image
    with shape (300, 300, 3), then the output will be an image with shape
    (height, width, 3).

  Raises:
    ValueError, if:
      - Input image's shape is not one of the required shapes below:
        * 2-D: [H, W]
        * 3-D: [H, W, C]
        * 4-D: [B, H, W, C]
  """
  if len(imgs.shape) not in [2, 3, 4]:
    raise ValueError(f'invalid img shape: {imgs.shape}')
  original_input_shape_len = len(imgs.shape)

  # add dims in order to make sure all inputs end up with same shape format.
  if len(imgs.shape) == 2:
    imgs = np.expand_dims(imgs, axis=-1)  # add channel
  if len(imgs.shape) == 3:
    imgs = np.expand_dims(imgs, axis=0)  # add batch

  b, h, w, c = imgs.shape
  if h != height or w != width:
    resized_imgs = np.empty(shape=(b, height, width, c), dtype=imgs.dtype)
    for i in range(b):
      img = imgs[i, :, :, :]
      # use OpenCV lib for resizing.
      # Note: Unlike ours, OpenCV's convention for shape is (w, h)
      img = cv2.resize(img, (width, height), interpolation=interpolation)
      if len(img.shape) == 2:  # OpenCV auto-reshapes grayscale
        img = np.expand_dims(img, axis=-1)
      resized_imgs[i, :, :, :] = img
  else:
    resized_imgs = imgs

  # squeeze back to original input shape, if needed
  if original_input_shape_len == 2:
    resized_imgs = np.squeeze(resized_imgs)  # squeeze all dims
  elif original_input_shape_len == 3:
    resized_imgs = np.squeeze(resized_imgs, axis=0)  # squeeze batch dim

  return resized_imgs

def normalize_img(imgs: np.ndarray) -> np.ndarray:
  """Normalizes input image to [0,1] range.

  The imgs arg can be either a single image or a batch of images, and is
  expected to be a uint8 array of any shape.

  Args:
    imgs: array representing input image(s)

  Returns:
    Array with normalized image(s) in range [0,1]. Please note that this
    function will change the image type to np.float32.

  Raises:
    ValueError, if input image is not of type uint8
  """
  if imgs.dtype != np.uint8:
    raise ValueError(
        f'Image array has Invalid type ({imgs.dtype}). Must be uint8')
  return (imgs / 255.).astype(np.float32)

def split_array_into_batches(arr: np.ndarray,
                             batch_size: int) -> List[np.ndarray]:
  """Splits array along the first dim to batches of batch_size.

  If arr is not evenly divisbile by batch_size, the last array in the returned
  list will have a length equal to len(arr) % batch_size.

  Args:
    arr: Array being split along its first dimension into batches of batch_size.
    batch_size: Size of batches to split arr into.

  Returns:
    A list of arrays.
  """
  return np.split(arr, np.arange(batch_size, len(arr), batch_size))


def load_detection_model(model_path, min_confidence_threshold: Optional[float], nms_threshold: Optional[float]):
    # Load the Centernet model
    model = tf.saved_model.load(model_path)
    model_fn = model.signatures['serving_default']
    input_info = model_fn.structured_input_signature[1]['input_tensor']
    input_key = input_info.name
    input_rank = input_info.shape.rank

    # Wrap the model with custom inference function
    def predict(img: np.ndarray) -> Dict[str, np.ndarray]:
      """Makes predictions on the image array.

      Args:
        img: [H, W, C] image array. Pixels can be [0,1] normalized or [0,255].

      Returns:
        Dictionary with the following key-value pairs:
          - 'boxes': [N, 4] float32 array of locations following this format:
                    [box_top, box_left, box_bottom, box_right]
          - 'classes': [N] float32 array of class labels for top predicted class.
          - 'scores': [N] float32 array of model confidence scores.
      """
      serialized_example = _np_img_to_serialized_tf_example(img)
      input_example = (
          serialized_example if input_rank == 0 else [serialized_example])
      saved_model_output = model(input_example)
      output_boxes = _convert_boxes_from_normalized_to_pixel_space(
          img, np.squeeze(saved_model_output[_DETECTION_BOXES],axis=0))

      return post_process_detections(output_boxes,
                                     np.squeeze(saved_model_output[_DETECTION_CLASSES],axis=0),
                                     np.squeeze(saved_model_output[_DETECTION_SCORES],axis=0),
                                     min_confidence_threshold, nms_threshold)


    return predict

def load_segmentation_model(model_path):
    model = tf.saved_model.load(model_path)
    # Perform any additional configuration based on your model's requirements

    """Raises errors if the model's inputs/outputs are invalid."""
    model_sigs = list(model.signatures.keys())
    if len(model_sigs) != 1:
      raise ValueError('Saved models should have exactly one model signature.')
    _model_signature_key = model_sigs[0]
    _model_fn = model.signatures[_model_signature_key]

    # Check and store model inputs and requirements.
    input_shape = list(_model_fn.inputs[0].shape)
    num_input_dims = len(input_shape)
    if num_input_dims != 4:
      raise ValueError('Saved model should expect 4-dim input '
                       f'(batch, height, width, 3). Got {num_input_dims} dims.')
    (_batch_size, _input_height, _input_width, _) = input_shape
    # Sometimes height and width are returned as a Dimension type object.
    # For certain calculations they are expected to be integers.
    _input_height = (
        int(_input_height) if _input_height else _input_height
    )
    _input_width = (
        int(_input_width) if _input_width else _input_width
    )
    if _batch_size is not None and _batch_size != 1:
      raise ValueError('Batch size must be 1 or None.'
                       f'Got {_batch_size}')
    _input_type = _model_fn.inputs[0].dtype
    if _input_type not in [tf.uint8, tf.float32]:
      raise ValueError('Expected input types are tf.uint8 and tf.float32. '
                       f'Got {_input_type}.')
    # Input will be normlized on [0,1] before passing to model if float32 is the
    # expected input type.
    _normalize_input = _input_type == tf.float32

    # Check and store model outputs and requirements.
    model_fn_output_keys = list(_model_fn.output_shapes.keys())
    # Ensure exactly one output key is in the list of expected keys.
    valid_keys_in_model = list(
        set(model_fn_output_keys).intersection(_VALID_OUTPUT_KEYS))
    if len(valid_keys_in_model) != 1:
      raise ValueError('Exactly one valid model output key not found in saved '
                       f'model. Expected one of: {_VALID_OUTPUT_KEYS}.'
                       f'Got: {model_fn_output_keys}.')
    _model_fn_output_key = valid_keys_in_model[0]
    output_shape = list(_model_fn.output_shapes[_model_fn_output_key])
    output_ndims = len(output_shape)
    if output_ndims != 3 and output_ndims != 4:
      raise ValueError('Saved model output tensor must be 3- or 4-dim.')
    _add_class_dim_to_output = False
    if output_ndims == 3:
      # Some u-net are single-class without a background class and don't have
      # a class dimension. A class dimensions will be added for these models.
      _add_class_dim_to_output = True
    _num_prediction_classes = output_shape[-1] if output_ndims == 4 else 1

    def _run_model(img: np.ndarray) -> np.ndarray:
      """Executes Tensorflow saved model on img."""
      img = tf.cast(img, _input_type)
      img_heatmaps = _model_fn(
          tf.convert_to_tensor(img))[_model_fn_output_key]
      np_img_heatmaps = img_heatmaps.numpy().astype(np.float32)
      if _add_class_dim_to_output:
        np_img_heatmaps = np.expand_dims(np_img_heatmaps, axis=-1)
      return np_img_heatmaps

    def predict(img: np.ndarray, batch_size: int = 8) -> np.ndarray:
      """Makes predictions on the image array(s).

      Args:
        img: [B, H, W, C] image array, where: B -> Batch H -> Image height W ->
          Image width C -> Image channels. If batch dimension is not present, it
          will be added.
        batch_size: Batch size sent to the underlying saved model.

      Returns:
        Array of inferred heatmaps with shape [B, H, W, C], where:
          B -> Batch
          H -> Image height
          W -> Image width
          C -> Prediction class

        If the model is learning N different classes, then C == N+1, where the
        first heatmap (index 0) represents the background class and the remaining
        N elements represent each of the N classes learned.

        NOTE 1: Each heatmap will have float32 values in the range [0, 1], where
                the value corresponts to the class probability of the given pixel.
                As a consequence, the sum of the values of a given pixel across
                all heatmaps of the list will add up to 1 (as the sum of
                probability across all classes must add up to 100%).

        NOTE 2: The output size will always correspond to the input image size,
                even if the underlying model accepts a different input size (the
                input image will be resized to match the saved model's
                expectations). For this reason, it's important that, if patching
                was used to train the model, the same patch sizes are fed into
                this method.
      """
      if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)  # add batch dimension.
      # Resize images if necessary.
      _, img_h, img_w, _ = img.shape
      should_resize_img = (
          (_input_height is not None and _input_width is not None)
          and (img_h != _input_height or img_w != _input_width))
      if should_resize_img:
        img = resize_img(
            imgs=img, width=_input_width, height=_input_height)

      # Normalize img pixels on [0, 1] if necessary.
      if _normalize_input:
        img = normalize_img(img)

      # Iterate over all imgs if batch size is fixed.
      all_heatmaps = []
      if _batch_size == 1:
        for single_img in img:
          single_img = np.expand_dims(single_img, axis=0)
          single_img_heatmaps = _run_model(single_img)
          all_heatmaps.append(single_img_heatmaps)
      # Run batches when possible.
      else:
        img_batches = split_array_into_batches(img, batch_size)
        for img_batch in img_batches:
          try:
            batch_heatmaps = _run_model(img_batch)
          except tf.errors.ResourceExhaustedError as e:
            raise ValueError(
                'Out of Memory Error. Batch size %s is likely too large. Try '
                'using a lower batch_size.', batch_size)
            raise e
          all_heatmaps.append(batch_heatmaps)
      output_heatmaps = np.concatenate(all_heatmaps, axis=0)
      # Resize back to original image dimensions if necessary.
      if resize_img:
        output_heatmaps = resize_img(
            imgs=output_heatmaps, width=img_w, height=img_h)

      return output_heatmaps


    return predict


def load_classification_model(model_path):
    classifier_model = tf.saved_model.load(model_path)
    classifier_model_fn = classifier_model.signatures['serving_default']
    # Perform any additional configuration based on your model's requirements
    return classifier_model, classifier_model_fn

def crop_image(img: np.ndarray, box: np.ndarray) -> np.ndarray:
  """Crop img based on box coordinates.

  Args:
    img: Image to be cropped. The first two dimensons should be indexable as
      (Y,X) coordinates.
    box: 4-element array of representing a box with following format [top_px,
      left_px, bottom_px, right_px].

  Returns:
    cropped image (using box as the crop region)
  """
  # Indexing requires ints. Some boxes are returned as floats even though
  # they're whole numbers.
  idx_box = box.copy()
  if not np.issubdtype(box.dtype, np.integer):
    idx_box = np.round(idx_box).astype(int)
  crop_top, crop_left, crop_bottom, crop_right = idx_box
  return img[crop_top:crop_bottom, crop_left:crop_right]

def get_gradability_prob(grain_head_img, gradability_model_fn, convert_to_bgr=True):
  # convert to BGR for wheat gradability filter
  grad_img = cv2.cvtColor(grain_head_img, cv2.COLOR_RGB2BGR) if convert_to_bgr else grain_head_img

  example_str = tf.train.Example(
      features=tf.train.Features(
          feature={'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_png(grad_img).numpy()]))
                  })).SerializeToString()
  output = gradability_model_fn(tf.convert_to_tensor(np.expand_dims(example_str, axis=0)))
  class_1_idx = np.argmax(np.squeeze(output['classes'].numpy()).astype(int))
  return np.squeeze(output['scores'].numpy())[class_1_idx]

def _is_mask_binarized(mask: np.ndarray) -> bool:
  """Returns True if mask is binarized with 0/1 or 0/255 values."""
  unique_vals = list(np.unique(mask))
  return unique_vals in [[0], [1], [255], [0, 1], [0, 255]]

def filter_mask_by_mask(
    input_mask: np.ndarray,
    filter_mask: np.ndarray,
    filter_mask_thresh: Optional[float] = None) -> np.ndarray:
  """Filters input_mask element-wise based on the values in filter_mask.

  If filter_mask values are 1/255 or exceed filter_mask_thresh, the values in
  input_mask are retained. If not, the input_mask values are set to 0. Common
  use-cases include:
    1. Filtering segmentation results using depth maps and vice-versa.
    2. Filtering results from one segmenter with the results from another (e.g.
       if you have separate segmenters that identify the crop area and count
       some feature of the plant, you may want to limit your counting results to
       only the identified crop area.

  Args:
    input_mask: N-dimensional mask to be filtered.
    filter_mask: Mask with same shape as input_mask used as element-wise filter.
      Unless filter_mask_thresh is provided, this array should be binarized with
      values of 0/255 or 0/1.
    filter_mask_thresh: Optional value by which filter_mask will be thresholded
      prior to filtering. This is often used as a convenience when a
      probabilistic mask (e.g. semantic segmentation output) is provided.

  Returns:
    Filtered input_mask.

  Raises:
    ValueError if:
      - input_mask and filter_mask are not of identical shape.
      - filter_mask doesn't contain only 0/255 or 0/1 when filter_mask_thresh
        not provided.
  """
  if input_mask.shape != filter_mask.shape:
    raise ValueError('Input and filter masks must have the same shape.')
  if filter_mask_thresh is None and not _is_mask_binarized(filter_mask):
    raise ValueError(
        'All values in mask be either 0 or 255 if filter_mask_thresh isn\'t '
        'provided.')

  binarized_filter = filter_mask.copy()
  # Ensure the binarized value is 255.
  if filter_mask_thresh is None:
    binarized_filter[binarized_filter != 0] = 255
  else:
    binarized_filter = np.where(filter_mask >= filter_mask_thresh, 255, 0)
  return np.where(binarized_filter == 255, input_mask,
                  0).astype(input_mask.dtype)

def resize_image_with_padding(image, height, width, text, text2):
  height_ratio = height / image.shape[0]
  width_ratio = width / image.shape[1]
  ratio = min(height_ratio, width_ratio)
  image_width = int(image.shape[1] * ratio + 0.5)
  image_height = int(image.shape[0] * ratio + 0.5)
  resized_image = cv2.resize(image, (image_width, image_height))
  resized_image = cv2.rectangle(resized_image, (0, 0), (image_width, image_height), (255, 255, 255), 2)
  if image_height < height:
    top = int((height - image_height)/2)
    bottom = height - image_height - top
    resized_image = cv2.copyMakeBorder(resized_image, top, bottom, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
  else:
    left = int((width - image_width)/2)
    right = width - image_width - left
    resized_image = cv2.copyMakeBorder(resized_image, 0, 0, left, width - image_width - left, cv2.BORDER_CONSTANT, (0, 0, 0))
  resized_image = cv2.copyMakeBorder(resized_image, 50, 0, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
  # Add first line of text
  resized_image = cv2.putText(resized_image, text, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, .75, (255,0,0), 2, cv2.LINE_AA)

# Add second line of text
  resized_image = cv2.putText(resized_image, text2, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, .75, (255,0,0), 2, cv2.LINE_AA)

  return resized_image

def overlay_mask_on_img(img: np.ndarray,
                        mask: np.ndarray,
                        mask_thresh: Optional[float] = None,
                        overlay_color: Tuple[int, int, int] = (255, 0, 0),
                        overlay_weight: float = 0.4) -> np.ndarray:
  """Overlays a translucent mask on an RGB image.

  The mask should be binarized with values 0/1 or 0/255 if mask_thresh is not
  provided. If mask_thresh is provided, the given mask will be binarized with
  foreground designated as all values in mask at or above mask_thresh.

  Args:
    img: (H,W,C) RGB image array.
    mask: (H,W) mask array containing only values 0/1 or 0/255 if mask_thresh is
      not provided.
    mask_thresh: Optional value by which mask will be thresholded. This is often
      used as a convenience when a probabilistic mask (e.g. semantic
      segmentation output) is provided.
    overlay_color: (R,G,B) tuple representing the color of the mask overlay.
    overlay_weight: Float on [0,1] indicating the percentage weight given to the
      mask overlain on the image. E.g. if the ovelay_weight is 0.4, the mask
      overlay will be given a weight of 40% while the original image will have a
      weight of 60%.

  Returns:
    Img with overlain translucent mask.

  Raises:
    ValueError if:
      - Arrays are of incorrect rank.
      - Mask doesn't contain only 0/1 or 0/255 when threshold not provided.
      - overlay_weight isn't on [0,1].
  """
  # Checks to ensure overlay will work properly.
  if img.ndim != 3:
    raise ValueError(
        f'img must be a rank 3 RGB matrix. Received rank {img.ndim}')
  if mask.ndim != 2:
    raise ValueError(f'mask must be a rank 2 matrix. Received rank {mask.ndim}')
  img_h, img_w, _ = img.shape
  mask_h, mask_w = mask.shape
  if mask_h != img_h or mask_w != img_w:
    raise ValueError('Image and mask must be the same size.')
  if mask_thresh is None and not _is_mask_binarized(mask):
    raise ValueError(
        'All values in mask be either 0 or 255 if mask_thresh isn\'t provided.')
  if overlay_weight < 0 or overlay_weight > 1:
    raise ValueError(
        f'overlay_weight must be on [0,1]. Received {overlay_weight}.')

  overlay_mask = mask.copy()
  # Ensure the binarized value is 255.
  if mask_thresh is None:
    overlay_mask[overlay_mask != 0] = 255
  else:
    overlay_mask = np.where(mask < mask_thresh, 0, 255)
  overlay_mask = cv2.cvtColor(overlay_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
  overlay_mask = np.where(overlay_mask == [255, 255, 255], overlay_color,
                          img).astype(np.uint8)
  return cv2.addWeighted(img, 1 - overlay_weight, overlay_mask, overlay_weight,
                         0)

def _prepare_image_for_concatenation(
    images: Sequence[np.ndarray], adjust_by_height: bool) -> List[np.ndarray]:
  """Pre-processes images for concatenation.

  This is achieved by resizing all images to have same width or height, and
  making sure the number of channels are the same (and equal to 3).
  Concatenation can be either vertical or horizontal, and can be specified by
  the `adjust_by_height` argument.

  Args:
    images: list of images to be processed
    adjust_by_height: if True, then all processed images will have the same
      height (meant for horizontal concatenation). If False, they will have the
      same width (meant for vertical concatenation).

  Returns:
    list of processed RGB images that have same width/height.

  Raises:
      ValueError if any of the image doesn't have 1, 3, or 4 channels.
  """
  max_dim = 0
  for img in images:
    if len(img.shape) < 2:
      raise ValueError('Invalid image array shape: {}'.format(img.shape))
    img_h = img.shape[0]
    img_w = img.shape[1]
    if adjust_by_height:
      max_dim = img_h if img_h > max_dim else max_dim
    else:  # adjust by width
      max_dim = img_w if img_w > max_dim else max_dim

  processed_imgs = []
  for img in images:
    # Fix the color if gray
    if len(img.shape) == 2:
      img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # Fix the color (if 4 channels)
    elif img.shape[-1] == 4:
      img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif img.shape[-1] not in {1, 3}:
      raise ValueError('Unexpected number of channels for image')

    # Resize image so that all images have the same height or width
    old_h, old_w, _ = img.shape
    old_dim = old_h if adjust_by_height else old_w
    resize_ratio = max_dim / old_dim
    if adjust_by_height:
      new_w = int(resize_ratio * old_w)
      new_h = max_dim
    else:
      new_w = max_dim
      new_h = int(resize_ratio * old_h)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    processed_imgs.append(img.astype(np.uint8))
  return processed_imgs

def horizontal_concatenation(images: Sequence[np.ndarray]) -> np.ndarray:
  """Concatenates a list of images horizontally (size-by-side).

  Args:
    images: list of images to be concatenated horizontally

  Returns:
    Single image composed by concatenating all input images horizontally
  """
  images = _prepare_image_for_concatenation(images, adjust_by_height=True)
  return cv2.hconcat(images)

def annot_img_with_fhb_disease(img, grain_head_hm, fhb_hm, grad_prob):
  fhb_in_head_heat = filter_mask_by_mask(fhb_hm, grain_head_hm, filter_mask_thresh=0.5)
  num_head_pixels = (grain_head_hm > 0.5).sum()
  num_fhb_in_head_pixels = (fhb_in_head_heat > 0.5).sum()
  perc_disease = round(num_fhb_in_head_pixels / num_head_pixels * 100, 2)
  grad = round(grad_prob, 2)
  annot_img = overlay_mask_on_img(img, grain_head_hm, mask_thresh=0.5, overlay_color=(0,0,255)) # Spike mask = BLUE
  annot_img = overlay_mask_on_img(annot_img, fhb_hm, mask_thresh=0.5, overlay_color=(255,0,0)) # Disease mask = RED
  annot_img = resize_image_with_padding(annot_img, 256, 256, f'FHB %: {perc_disease}', f'Grad: {grad}')
  orig_img = resize_image_with_padding(img, 256, 256, 'Original', '')
  fhb_annotated_image = horizontal_concatenation([orig_img, annot_img])
  return horizontal_concatenation([orig_img, annot_img])

def pick_rgb_color_from_palette(color_idx: int) -> Tuple[int, int, int]:
  """Picks a RGB color from a pre-defined palette.

  This function was created with two objectives:
  - a given color_idx will always receive the same color
  - colors are spaced out in a way where you get a good distribution and
    diversity of colors, no matter how many color indexes you have.

  Args:
    color_idx: index of the color to be picked from a pre-defined palette

  Returns:
    3-d int tuple representing the RGB color picked, e.g. (10, 135, 31).

  Raises:
      ValueError if:
      - color_idx is not in interval [0, +inf]
  """
  if color_idx < 0:
    raise ValueError('color_idx must be in interval [0, max_idx]')

  # for the first 20 colors, we have some hard-coded set of clear and distinct
  # colors. For most usecases, we expect less than 20 colors, so this should
  # be a good set of initial colors. If more colors are needed, we'll use a
  # hashing approach as a deterministic proxy to select colors in a way that is
  # approximately uniform in the color space.
  start_palette = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
                   (245, 130, 48), (145, 30, 180),
                   (70, 240, 240), (240, 50, 230), (210, 245, 60),
                   (250, 190, 212), (0, 128, 128), (220, 190, 255),
                   (170, 110, 40), (255, 250, 200), (128, 0, 0),
                   (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128),
                   (128, 128, 128)]

  # use the pre-established selection of distinct colors above
  if color_idx < len(start_palette):
    return start_palette[color_idx]
  # else, get a deterministic (but arbitrary color selection) based on hashing
  m = hashlib.md5()
  m.update(f'red_{color_idx}'.encode('utf-8'))
  red = int(m.hexdigest(), 16) % 256
  m.update(f'green_{color_idx}'.encode('utf-8'))
  green = int(m.hexdigest(), 16) % 256
  m.update(f'blue_{color_idx}'.encode('utf-8'))
  blue = int(m.hexdigest(), 16) % 256
  return (red, green, blue)

def draw_bounding_box_predictions(
    img: np.ndarray,
    bb_predictions: Dict[str, Any],
    thickness: Optional[int] = 3,
    colors: Optional[Sequence[Tuple[int, int, int]]] = None) -> np.ndarray:
  """Draws bounding box model predictions on an image.

  Args:
    img: image on which we want to draw the bounding boxes. Notice that this
      image won't be modified. It will be cloned and the bounding boxes will be
      drawn on top of that clone.
    bb_predictions: bounding box predictions using the standard format of object
      detection inference output: a dictionary containing the following keys:
        ['boxes', 'classes', 'scores']
    thickness: optional bounding box thickness.
    colors: optional list of colors to use, one per class. If not provided,
      colors will be picked automatically.

  Returns:
    Image annotated with bounding boxes.

  Raises:
      ValueError if:
      - bb_predictions doesn't contain key 'boxes'
      - bb_predictions doesn't contain key 'classes'
      - 'boxes' and 'classes' are not the same length
      - if colors is provided but not with enough colors to cover all classes
  """
  if 'boxes' not in bb_predictions:
    raise ValueError(f'bb_predictions must contain \'boxes\' key.'
                     f'Current keys:{bb_predictions.keys()}')
  if 'classes' not in bb_predictions:
    raise ValueError(f'bb_predictions must contain \'classes\' key.'
                     f'Current keys:{bb_predictions.keys()}')
  if len(bb_predictions['boxes']) != len(bb_predictions['classes']):
    raise ValueError('boxes and classes must have the same number of elements.')

  if not bb_predictions['classes'].size:
    return np.array(img, copy=True)

  n_classes = int(np.max(bb_predictions['classes'])) + 1  # can be 0-indexed

  if colors and len(colors) < n_classes:
    raise ValueError(
        f'Colors arg was provided, but not enough colors for all classes.'
        f'# provided colors:{len(colors)}. # classes: {n_classes}')

  annotated_img = np.array(img, copy=True)
  for box, bb_class in zip(bb_predictions['boxes'], bb_predictions['classes']):
    y1, x1, y2, x2 = box.astype(int)
    bb_class = int(bb_class)
    color = pick_rgb_color_from_palette(
        bb_class) if colors is None else colors[bb_class]
    annotated_img = cv2.rectangle(
        annotated_img, (x1, y1), (x2, y2), color=color, thickness=thickness)

  return annotated_img

# Saves side-by-side of single spikes from each full rover image, original and with spike and disease masks
# Change annot_img_dir to save annotated images
def run_fhb_pipeline(img_path, 
                     head_detector, 
                     head_segmenter, 
                     fhb_segmenter, 
                     gradability_fn, 
                     annot_img_bool=True, 
                     #annot_img_dir=f(output_directory, 'pipeline/'), 
                     convert_to_bgr=True, 
                     bb_coords=False, 
                     detection_rotation=90, # Changed from 0 (v2) to 90 (v3)
                     segmentation_rotation=-90, # Changed from 0 (v2) to -90 (v3)
                     fhb_rotation=0, 
                     gradability_rotation=0, # Changed from 0 (v2) to 90 (v3). Did not work. Changing back to 0 3/26/24 @ 11:44 AM
                     visualize_samples=True):
  
  ## V1 Original
  # image is rotated ccw 90 degrees
  # img = rotate(cv2.imread(img_path), detection_rotation)

  ## V2 Changed 3/21/24
  # image is rotated ccw 90 degrees
  bgr_img = rotate(cv2.imread(img_path), detection_rotation)
  img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

  # these are applied to non_rotated images (do not rotate)
  if bb_coords is True:
    img_path = img_path.split('.')[0]
    coords_str = img_path.split('--')[1]
    grain_boxes = np.array([coords_str.split('_')])
    img_crop =crop_image(img, np.array([int(coord) for coord in grain_boxes[0]]))
    # plt.imshow(img_crop)
    grain_head_crops = [img_crop]

  else:
    head_dets = head_detector(img)
    grain_boxes = head_dets['boxes'][head_dets['classes'] == 1]
    grain_head_crops = [crop_image(img, box) for box in grain_boxes]

  grain_head_segmentations = [rotate(head_segmenter(rotate(head_crop, segmentation_rotation)).squeeze()[:,:,1], -segmentation_rotation) for head_crop in grain_head_crops]

  grain_fhb_segmentations = [rotate(fhb_segmenter(rotate(head_crop, fhb_rotation)).squeeze()[:,:,1], -fhb_rotation) for head_crop in grain_head_crops]

  head_gradability_probs = [get_gradability_prob(rotate(head_crop, gradability_rotation), gradability_fn, convert_to_bgr) for head_crop in grain_head_crops]
  
  fhb_in_head_heatmaps = [filter_mask_by_mask(fhb_heat, grain_head_heat, filter_mask_thresh=0.5)
                          for fhb_heat, grain_head_heat in zip(grain_fhb_segmentations, grain_head_segmentations)]
  num_head_pixels = [(grain_head_heat > 0.5).sum() for grain_head_heat in grain_head_segmentations]
  num_fhb_pixels_in_head = [(fhb_heat > 0.5).sum() for fhb_heat in fhb_in_head_heatmaps]
  # [0,1]
  perc_fhb_in_head = [fhb_pix / head_pix for fhb_pix, head_pix in zip(num_fhb_pixels_in_head, num_head_pixels)]
  num_fhb_pixels = [(fhb_heat > 0.5).sum() for fhb_heat in grain_fhb_segmentations]
  # [0,1]
  perc_fhb_pix_in_img = [fhb_pix / fhb_seg.size for fhb_pix, fhb_seg in zip(num_fhb_pixels, grain_fhb_segmentations)]
  # [0,100]
  fhb_percentage = [decimal_fhb * 100 for decimal_fhb in perc_fhb_in_head]
  fhb_df = pd.DataFrame({
    'num_fhb_pixels': num_fhb_pixels,
    'num_head_pixels': num_head_pixels,
    'num_fhb_pixels_in_head': num_fhb_pixels_in_head,
    'perc_fhb_in_head': perc_fhb_in_head,
    'perc_fhb_pix_in_img': perc_fhb_pix_in_img,
    'gradability_prob': head_gradability_probs,
    'fhb_percentage': fhb_percentage,
    'box_coords': grain_boxes.tolist()
  })
  fhb_df['img_path'] = img_path
  # top, left, bottom, right
  bbox_df = pd.DataFrame(grain_boxes, columns=['y_start', 'x_start', 'y_end', 'x_end'])
  fhb_df = pd.concat([fhb_df, bbox_df], axis=1)
  
# Iterate over each unique box_coords and save the annotated image
  for idx, (box, img, grain_head_hm, fhb_hm, grad_prob) in enumerate(zip(grain_boxes, grain_head_crops, grain_head_segmentations, grain_fhb_segmentations, head_gradability_probs)):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        annot_img = annot_img_with_fhb_disease(img, grain_head_hm, fhb_hm, grad_prob)
        annot_img = cv2.cvtColor(annot_img, cv2.COLOR_BGR2RGB)
        #if annot_img_dir is not None:
        save_img_path = os.path.join(spike_dir, f"{img_name}_box{idx}.png")
        cv2.imwrite(save_img_path, annot_img, [cv2.IMWRITE_PNG_COMPRESSION, 0]) # No compression, best quality
  return fhb_df


# Print original image and number of grain heads detected
def side_by_side_sampled_images(sampled_file_paths, object_detector,
                                border_y_val=None, rotation=0):
  side_by_side_img = []
  side_by_side_img_bb = []
  num_grain_heads_list = []
  for i,_ in enumerate(sampled_file_paths):
    img = rotate(cv2.imread(img_path), rotation)
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    side_by_side_img.append(img)

    img_annotate = img.copy()
    od_pred = object_detector(img_annotate)
    #print(od_pred)
    img_annotate = draw_bounding_box_predictions(img_annotate,
                                                                      bb_predictions=od_pred,
                                                                      colors=[None, 
                                                                              (0, 0, 255), # Red for wheat spikes
                                                                              (255, 0, 0)]) # Blue
    print(od_pred['classes'])
    num_grain_heads = len(od_pred['boxes'][od_pred['classes'] == 1])
    num_other_heads = len(od_pred['boxes'][od_pred['classes'] == 0])
    print('count grain {}: {}'.format(i, num_grain_heads))
    print('count other {}: {}'.format(i, num_other_heads))

    num_grain_heads_list.append(num_grain_heads)

    if border_y_val is not None:
      border_line_color = (0,0,255)
      x1,x2=0,2048
      y1=border_y_val
      y2=y1
      cv2.line(img_annotate, (x1,y1), (x2,y2), border_line_color, thickness=3)

    side_by_side_img_bb.append(img_annotate)

  side_by_side_original= horizontal_concatenation(side_by_side_img)
  side_by_side_annotate = horizontal_concatenation(side_by_side_img_bb)
  # Concatenate images side by side
  concatenated_image = np.hstack((side_by_side_original, side_by_side_annotate))
  # Save the concatenated image
  #save_img_path = os.path.join(bounding_box_dir, f"{img_name}.png")
  #cv2.imwrite(save_img_path, concatenated_image, [cv2.IMWRITE_PNG_COMPRESSION, 0]) # No compression, best quality
  return side_by_side_original, side_by_side_annotate, num_grain_heads_list

timestamp_proto_definition = """
syntax = "proto2";

message Timestamp {
  // Represents seconds of UTC time since Unix epoch
  // 1970-01-01T00:00:00Z. Must be from 0001-01-01T00:00:00Z to
  // 9999-12-31T23:59:59Z inclusive.
  optional int64 seconds = 1;

  // Non-negative fractions of a second at nanosecond resolution. Negative
  // second values with fractions must still have non-negative nanos values
  // that count forward in time. Must be from 0 to 999,999,999
  // inclusive.
  optional int32 nanos = 2;
}
"""

with open('timestamp.proto', 'w') as f:
  f.write(timestamp_proto_definition)

#!protoc --python_out=. timestamp.proto
#import timestamp_pb2
# Cory modified code to run outside of jupyter notebook
subprocess.run(['protoc', '--python_out=.', 'timestamp.proto'])

data_registration_proto_definition = """
syntax = "proto2";

import "timestamp.proto";

message GeoPoint {
  optional double longitude = 1;
  optional double latitude = 2;
}

// Identifies a rectangle area.
message GeoRectangle {
  optional GeoPoint top_left = 1;
  optional GeoPoint top_right = 2;
  optional GeoPoint bottom_right = 3;
  // optional
  optional GeoPoint bottom_left = 4;
}


message RegisteredImage {
  // The data point's file path.
  optional string file_path = 1;

  // The time when this file(image) is captured.
  optional Timestamp capture_time = 7;

  // Unique identifies the row/sub-field the image is located.
  optional string field_name = 2;
  optional string row_name = 3;
  optional string plot_name = 4;
  optional string entry_name = 15;
  optional string admin_level = 8;
  optional string session_id = 11;

  // Final registered location, after compensated the drift.
  optional GeoRectangle corners = 5;

  // The rtk geo point from localization.
  optional GeoPoint rtk_geo_point = 12;

  // The leaf area index of this image. Used to perform day-to-day alignment.
  optional double leaf_area_index = 6 [deprecated = true];

  // A list of image signals. e.g. Height, LEAF_INDEX, etc
  map<string, double> image_signals = 9;

  // If set, only the part indicated by this bounding boxes are registered.
  repeated bytes partial_image_boxes = 10;

  // In the context of edge computing, indicates whether the image is
  // intended to be uploaded. Note that the image might be dropped for a
  // variety of reasons on the edge side, and edge is responsible for populating
  // some signals (i.e. set ImageUploadIntent).
  optional bool image_upload_intent = 13;

  // The yaw of the truckee at the time of photo capture, expressed in radians.
  // The origin is at true east and moving counter clock wise increases the
  // radians. The yaw follows the "right hand rule" where your thumb points up.
  // The yaw range is [pi, -pi) with true east yaw = 0, north = pi/2,
  // south = -pi/2 and west = pi.
  optional float yaw_rad = 14;
}
message RegisteredImageList {
  // The list of objects.
  repeated RegisteredImage images = 1;
  // Name of the camera.
  optional string camera_name = 2;
  // direction of the row, used for rotated patches and stitched result.
  // direction of the row, used for rotated patches and stitched result.
  optional bytes direction = 3;
  // Orientation of the images in RegisteredImageList,
  // compared to the original orientation.
  // This field is used to capture image rotation during Bagfile Extraction.
  // The orientation is homogenous across the same session and camera.
  optional bytes rotation = 4;
}

"""


with open('data_registration.proto', 'w') as f:
  f.write(data_registration_proto_definition)

#!protoc --python_out=. data_registration.proto
#import data_registration_pb2
# Cory added code to work outside of jupyter notebook
# Execute the protoc command using subprocess
subprocess.run(['protoc', '--python_out=.', 'data_registration.proto'])

# parse data registration
def parse_registration_file(path, camera):
  _REGISTERED_IMAGES_DF_COLUMNS = ['admin_level', 'filepaths_{}'.format(camera), 'session_id', 'rtk_long', 'rtk_lat', 'entry', 'row', 'plot',]
  registered_images_dict = dict()
  with open(path, 'rb') as fp:
    #registered_images_list = data_registration_pb2.RegisteredImageList()
    #registered_images_list.ParseFromString(fp.read())

    registered_images_list = text_format.Parse(fp.read(), data_registration_pb2.RegisteredImageList())
    # registered_images_list = message_type.ParseFromString(fp.read())
    for registered_image in registered_images_list.images:
      registered_images_dict[f'{os.path.basename(registered_image.file_path)}-{registered_image.row_name}'] = [
          registered_image.admin_level,
          registered_image.file_path,
          registered_image.session_id,
          registered_image.rtk_geo_point.longitude,
          registered_image.rtk_geo_point.latitude,
          -10 if not registered_image.entry_name else int(registered_image.entry_name),
          -10 if not registered_image.row_name else int(registered_image.row_name),
          -10 if not registered_image.plot_name else int(registered_image.plot_name)]
  return pd.DataFrame.from_dict(registered_images_dict, columns=_REGISTERED_IMAGES_DF_COLUMNS, orient='index')

def load_from_registered_images_filelist(registered_images_files, camera, customer_folder):
  df=pd.DataFrame()
  for filepath in registered_images_files:
    task_df = parse_registration_file(filepath, camera)
    df = pd.concat([df, task_df], ignore_index=True)
  # data formatting: assign row and plot to each image
  df['row_plot'] = df.apply(lambda row: 'r{}/p{}'.format(row['row'], row['plot']), axis=1)
  df['tmp_path'] = df['filepaths_{}'.format(camera)].apply(lambda x: ('/').join(x.split('/')[:8]))

  # replace path with bigstore
  df['filepaths_{}'.format(camera)] = df.apply(lambda row: row['filepaths_{}'.format(camera)].replace(row.tmp_path, customer_folder), axis=1)
  return df

alta_proto_definition = """
syntax = "proto2";

import "timestamp.proto";

message Trip {
  // Contains the link to the filepath for the raw gnss data.
  message GnssObject {
    optional string file_path = 1;
    optional Timestamp create_time = 2;
    optional Timestamp end_time = 3;
  }

  // Trip state.
  enum State {
    STATE_UNSPECIFIED = 0;
    FINISHED = 1;
  }

  optional State state = 1;
  optional Timestamp create_time = 2;
  optional Timestamp end_time = 5;

  // The tasks contained in this trip.
  repeated Task tasks = 3;

  // Gnss data.
  optional GnssObject gnss_log = 4;
}

// A task is a single data collection event.
// Next id: 11
message Task {
  // Task descriptors.
  optional string task_uuid = 1;
  optional string task_name = 2;

  // Campaign information.
  optional bytes campaign = 3;

  // A list of plot boundaries marked by timestamp.
  repeated Timestamp plot_boundaries = 4;

  // When this task was created and completed.
  optional Timestamp create_time = 5;
  optional Timestamp end_time = 9;

  // Forward-slash delineated string of this specific location.
  optional string admin_level = 6;

  // The images taken on this task.
  repeated ImageObject images = 7;

  // The videos taken on this task.
  repeated VideoObject videos = 10;

  // Store information of parent trip, if necessary.
  optional string trip_uuid = 8;
}

message GroundTruthValue {
  oneof value {
    string str_val = 1;
    int64 int_val = 2;
    float float_val = 3;
    bool bool_val = 4;
  }
}

message BoundingBox2D {
  // The four fields below are the bounds of the box, pixels.
  optional float left = 1;
  optional float top = 2;
  optional float right = 3;   // Exclusive: [left, right)
  optional float bottom = 4;  // Exclusive: [top, bottom)

  optional float confidence = 5;

  optional int32 detection_class = 6;

  // Name of detection class, it is optional.
  // Check photos/vision/object_detection/mobile/research/tools/label_map.proto.
  optional string name = 7;

  // Display name of detection class, it is optional.
  // Check photos/vision/object_detection/mobile/research/tools/label_map.proto.
  optional string display_name = 8;

  // Uncertainty metric associated with the box classification, it is optional.
  // 1 - (Max Class Probability - 2nd Highest Max Class Probability)
  // The higher this is, the higher the estimated uncertainty of prediction
  optional float margin_uncertainty = 9;

  // A list of labels for bounding box level ground truth.
  map<string, GroundTruthValue> gt_labels = 10;
}

message Point {
  // Latitude in degrees, in the range [-90, 90].
  required double latitude = 1;

  // Longitude in degrees, in the range [-180, 180].
  required double longitude = 2;

  // Optional altitude in meters relative to some vertical datum.
  // Positive values indicate a point above the datum. The choice of
  // vertical datum is left up to the application, since Spanner
  // doesn't use altitude data in any way. (Spanner performs all
  // geometric operations using only the latitude and longitude
  // values.) Picking Mean Sea Level as the datum is a good choice,
  // but other choices are possible. See the design document
  // (link at top of file) for more information.
  optional double altitude = 3;
}

// An image object is an image, along with annotation and inference boxes.
message ImageObject {
  optional string image_path = 1;

  // Timestamp of image creation.
  optional Timestamp create_time = 2;

  // A list of image labels from in-app annotation for image-level groundtruth.
  map<string, GroundTruthValue> gt_img_labels = 10;

  // A list of boxes from in-app annotations for box-level groundtruth.
  repeated BoundingBox2D gt_labels = 3;

  // A list of predicted boxes from online inference.
  repeated BoundingBox2D pred_labels = 4;

  // Reference to parent task.
  optional string task_uuid = 5;

  // NIMA score.
  optional float nima_score = 6;

  // Free text description of the image, entered by the user.
  optional string text_description = 7;

  // GPS point.
  optional Point point = 8;

  // Image object tags.
  // For Alta Data Collection, tags should start with 'ADC_',
  // For Alta Scouting, tags should start with 'AS_'.
  enum ImageObjectTag {
    UNSPECIFIED = 0;
    ADC_DEFAULT_MULTIPLE_PLANTS = 1;
    ADC_CLOSE_UP_SINGLE_PLANT = 2;
    ADC_UNDER_CANOPY_SINGLE_PLANT = 3;
    ADC_UNDER_CANOPY_MULTIPLE_PLANTS = 7;
    ADC_TOP_DOWN_WHOLE_PLOT = 4;
    ADC_TOP_DOWN_MULTIPLE_PLANTS = 5;
    ADC_TOP_DOWN_PANORAMA = 6;
    // Take images at approximately 45 degrees to the ground.
    ADC_TILT_SHOT = 8;
  }

  // A tag that defines the image characteristic such as shooting angle, object
  // being shot etc.
  optional ImageObjectTag tag = 9;

  // Information of the edge model bundle that produced the pred_labels. This
  // field is only set when pred_label exist. Only subfields model_bundle_id and
  // type will have value.
  optional bytes
      edge_model_bundle_info = 11;

  // Computed device orientation from phone's position sensors.
  // Reference source:
  // https://developer.android.com/guide/topics/sensors/sensors_position#sensors-pos-orient
  optional ImuEventData device_orientation = 13;
  reserved 12;
}

// A data point for imu readings, e.g. gyroscope, accelerometer, magnetometer.
message ImuEventData {
  optional double x = 1;
  optional double y = 2;
  optional double z = 3;
}

// A VideoObject is a video taken in apps.
message VideoObject {
  optional string video_path = 1;

  // Timestamp of image creation.
  optional Timestamp create_time = 2;

  // The duration of the video in seconds.
  optional int64 duration_seconds = 3;
}

"""

with open('alta.proto', 'w') as f:
  f.write(alta_proto_definition)

#!protoc --python_out=. alta.proto
#import alta_pb2
# Cory modified code to work outside of jupyter notebook
# Execute the protoc command using subprocess
subprocess.run(['protoc', '--python_out=.', 'alta.proto'])

def get_trip_files(trip_dir, file_filter=None):
  file_paths = []
  for file_or_dir in tf.io.gfile.listdir(trip_dir):
      full_path = tf.io.gfile.join(trip_dir, file_or_dir)
      if tf.io.gfile.isdir(full_path):
          file_paths.extend(get_trip_files(full_path, file_filter))
      else:
          if file_filter:
            if full_path.endswith(file_filter):
              file_paths.append(full_path)
  return file_paths


def trip_pb_to_df(trip_pb_file):
  """Obtain trip information as a pandas dataframe accounting for presence of visual guides.

  trip_dir: trip folder which includes trip_id belonging to a user.
  image_path_base: path with format '/bigstore/agdata-agility/edge/<user_email>/alta/'
  """

  with tf.io.gfile.GFile(trip_pb_file, 'rb') as fp:
    alta_trip = alta_pb2.Trip()
    trip_msg = alta_trip.FromString(fp.read())

  _ALTA_TRIPS_COLUMNS = ['admin_level','image_path','visual_guide_type','top','left','bottom','right']
  trip_dict = dict()
  idx = 0
  for task in trip_msg.tasks:
    admin_level = task.admin_level
    for image in task.images:
      if len(image.gt_labels) > 0:
        trip_dict[idx] = [admin_level, image.image_path,
                                image.gt_labels[0].name,
                                image.gt_labels[0].top,
                                image.gt_labels[0].left,
                                image.gt_labels[0].bottom,
                                image.gt_labels[0].right]
      else:
        # no visual guide
        trip_dict[idx] = [admin_level, image.image_path,
                                  "",
                                  "",
                                  "",
                                  "",
                                  ""]
      idx+=1
  trip_df = pd.DataFrame.from_dict(trip_dict, columns=_ALTA_TRIPS_COLUMNS, orient='index')
  trip_df['image_path'] = trip_df['image_path'].apply(lambda x: image_path_base + x)
  return trip_df

# Load Models

# Configurable parameters: fhb/models/models_2021
grain_head_object_detection_model_path = "/Users/jcooper/Desktop/thesis_research/fhb_umn/models/models_2021/grain_head_and_other_detection/umn_full_growth_cycle_centernet_1024x1024" # @param {type:"string"}
label_map_path = "/Users/jcooper/Desktop/thesis_research/fhb_umn/models/models_2021/grain_head_and_other_detection/label_map.pbtxt" # @param {type:"string"}
grain_head_segmentation_model_path = "/Users/jcooper/Desktop/thesis_research/fhb_umn/models/models_2021/umn_full_growth_cycle_unet_128x128" # @param {type:"string"}
wheat_head_fhb_disease_segmentation_model_path = "/Users/jcooper/Desktop/thesis_research/fhb_umn/models/models_2021/wheat_head_fhb_seg/wheat_fhb_instance_seg_data_wo_disease_256x256" # @param {type:"string"}
wheat_head_fhb_disease_gradability_model_path = "/Users/jcooper/Desktop/thesis_research/fhb_umn/models/models_2021/wheat_spike_fhb_disease_non_gradable_classification_input_is_BGR/resnet101x3_dnn230_212" # @param {type:"string"}

min_confidence_threshold = 0.5  # @param {type:"int"} # Set your desired minimum confidence threshold
nms_overlap_threshold = 0.3  # @param {type:"int"} # Set your desired NMS overlapping threshold

# Load models
grain_head_detector = load_detection_model(grain_head_object_detection_model_path, min_confidence_threshold=0.3, nms_threshold=0.7)
grain_head_segmenter = load_segmentation_model(grain_head_segmentation_model_path)
wheat_fhb_segmenter = load_segmentation_model(wheat_head_fhb_disease_segmentation_model_path)
wheat_gradability_classifier, wheat_gradability_classifier_fn = load_classification_model(wheat_head_fhb_disease_gradability_model_path)

#################################
#################################
#################################

"""Input the img_path for the image that you'd like to inference on"""

img_path = "/Users/jcooper/Desktop/fhb_pipeline_test/camera_tests/20240718/PXL_20240718_164845173/PXL_20240718_164845173.jpg"
img_path_name_only = re.search(r'/([^/]+)\.jpg$', img_path).group(1) # Need to change based on file type

# Get the output directory from the command-line arguments
output_directory = "/Users/jcooper/Desktop/fhb_pipeline_test/camera_tests/20240718/PXL_20240718_164845173"

# Create the output directory if it does not exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    spike_dir = os.path.join(output_directory, "spikes")
    #bounding_box_dir = os.path.join(output_directory, "bounding_box")
    os.makedirs(spike_dir)
    #os.makedirs(bounding_box_dir)
    print(f"Created directory: {output_directory}")
else:
    spike_dir = os.path.join(output_directory, "spikes")
    #bounding_box_dir = os.path.join(output_directory, "bounding_box")
    print(f"Directory already exists: {output_directory}")

#################################
#################################
#################################

"""# End-to-end Inference and Visualization

Run this cell to get a segmentation overlay mask of each detected wheat head from the image with a %FHB score.
"""

plt.rcParams['figure.figsize'] = (10/2, 12/4) # set display size
curr_df = run_fhb_pipeline(img_path=img_path,
                        head_detector=grain_head_detector,
                        head_segmenter=grain_head_segmenter,
                        fhb_segmenter=wheat_fhb_segmenter,
                        gradability_fn=wheat_gradability_classifier_fn,
                        convert_to_bgr=True, # changed from False on 3/27. More gradable spikes when False. Results match Mineral. 
                        detection_rotation=90, # Vertical
                        segmentation_rotation=-90, # Horizontal
                        fhb_rotation=0, # Vertical, changed from 90 (v2) to 0 (v3)
                        gradability_rotation=0, 
                        visualize_samples=True
                                    )

#If the grain heads are initially horizontal in the full image, we first apply rotation on the full image
#detection_rotation=90 rotates the image 90 deg ccw (img = rotate(pipe_utils.read_image(img_path), detection_rotation))
#each grain_head_crops is vertical
#in grain_head_segmentations, we apply segmentation_rotation from vertical to horizontal, so we use segmentation_rotation=-90  (but this is undone at the end so the head goes back to vertical)
#since after segmentation, the head is vertical again, we have fhb_rotation=0


# Construct the full path for the results CSV file
csv_file_path = os.path.join(output_directory, f"{img_path_name_only}_data_frame.csv")
# Save the results DataFrame to the CSV file
curr_df.to_csv(csv_file_path, index=False)


"""Detect Heads and Visualize"""

plt.rcParams['figure.figsize'] = (10*2, 12) # set display size
plt.axis('off')
# apply 90 ccw rotation as the head detection model was only trained with vertical heads truckee images in 2022 were oriented horizontally throughout the season.
original, annotated, num_bb = side_by_side_sampled_images([img_path], grain_head_detector, rotation=90)
detect_heads_and_visualize = horizontal_concatenation([original, annotated])
save_img_path = os.path.join(output_directory, f"{img_path_name_only}_detect_heads_and_visualize.png")
cv2.imwrite(save_img_path, detect_heads_and_visualize, [cv2.IMWRITE_PNG_COMPRESSION, 0]) # No compression, best quality

"""Disease heatmap visualization"""
# Add row number to the DataFrame
curr_df['row_number'] = curr_df.index

# Compute centroids
curr_df['centroid']=curr_df['box_coords'].apply(lambda box: [(int(box[1])+int(box[3]))//2, (int(box[0])+int(box[2]))//2])

# load original image
rotation = 90
gradability_threshold = 0.5 # 1 = ungradable, 0 = gradable

## V2 Changed 3/21/24
bgr_img = rotate(cv2.imread(img_path), rotation)
original_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

# get gradable and ungradable detections associated with the image
inference_img_df = curr_df.copy()
gradable_df = inference_img_df[inference_img_df['gradability_prob'] <= gradability_threshold]
ungradable_df = inference_img_df[inference_img_df['gradability_prob'] > gradability_threshold]

inference_img_bb_gradable = gradable_df[gradable_df['img_path']==img_path].box_coords.tolist()
inference_img_bb_ungradable = ungradable_df[ungradable_df['img_path']==img_path].box_coords.tolist()

# annotate detection of gradable and ungradable heads
img_annotated = draw_bounding_box_predictions(original_img,
                                                        bb_predictions={'boxes':np.array(inference_img_bb_gradable),
                                                                        'classes':np.array(np.ones(len(inference_img_bb_gradable)))},
                                                                      colors=[None, 
                                                                              (255, 0, 0), 
                                                                              (0, 0, 255)])
# Check how many ungradable heads were detected
print(len(inference_img_bb_ungradable))
img_annotated = draw_bounding_box_predictions(img_annotated,
                                                                    bb_predictions={'boxes':np.array(inference_img_bb_ungradable),
                                                                                    'classes':np.array(np.ones(len(inference_img_bb_ungradable)))},
                                                                    colors=[None, (0, 0, 0), (0, 0, 0)])

# visualize disease
annotated_disease_img = np.array(original_img, copy=True)
# draw ungradable
annotated_disease_img = draw_bounding_box_predictions(annotated_disease_img,
                                                                    bb_predictions={'boxes':np.array(inference_img_bb_ungradable),
                                                                                    'classes':np.array(np.ones(len(inference_img_bb_ungradable)))},
                                                                    colors=[None, (0, 0, 0), (0, 0, 0)])


cmap = matplotlib.cm.YlOrRd
norm = matplotlib.colors.Normalize(vmin=0, vmax=100)

# labeled heads
for box, fhb_val, centroid, row_number in zip(gradable_df['box_coords'], gradable_df['fhb_percentage'], gradable_df['centroid'], gradable_df['row_number']):
    box = np.array([int(x) for x in box])
    y1, x1, y2, x2 = box
    fhb_perc = int(fhb_val)
    color = cmap(norm(fhb_perc))
    color_transparent = list(color)
    color_transparent = [int(c * 255) for c in color_transparent]
    color_transparent = color_transparent[:3]
    color = tuple(color_transparent)

    sub_img = annotated_disease_img[y1:y2, x1:x2]
    rectangle = cv2.rectangle(sub_img.copy(), (0,0), (x2-x1, y2-y1), color, -1)

    res = cv2.addWeighted(sub_img, 0.5, rectangle, 0.5, 1)
    annotated_disease_img[y1:y2, x1:x2] = res

 # Annotate with FHB percentage and row number
    annotation_text_fhb = f"{fhb_perc}%"
    annotation_text_row = f"({row_number})"
    cv2.putText(annotated_disease_img, annotation_text_fhb, (int(centroid[0]), int(centroid[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    cv2.putText(annotated_disease_img, annotation_text_row, (int(centroid[0]), int(centroid[1]) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
side_by_side_img = horizontal_concatenation([original_img, img_annotated, annotated_disease_img])
side_by_side_img = cv2.cvtColor(side_by_side_img, cv2.COLOR_BGR2RGB)
save_img_path = os.path.join(output_directory, f"{img_path_name_only}_side_by_side.png")
cv2.imwrite(save_img_path, side_by_side_img, [cv2.IMWRITE_PNG_COMPRESSION, 0]) # No compression, best quality
                

# Plot seperation
# Method: DBSCAN
# Load original image
rotation = 90
gradability_threshold = 0.5  # 1 = ungradable, 0 = gradable
bgr_img = rotate(cv2.imread(img_path), rotation)
original_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

# Get gradable and ungradable detections associated with the image
inference_img_df = curr_df.copy()
gradable_df = inference_img_df[inference_img_df['gradability_prob'] <= gradability_threshold].copy()

# Calculate centroid
gradable_df['centroid'] = gradable_df['box_coords'].apply(lambda box: [(int(box[1]) + int(box[3])) // 2, (int(box[0]) + int(box[2])) // 2])
centroid = np.array(gradable_df['centroid'].tolist())

# Apply DBSCAN clustering
epsilon = 500  # Adjust based on your specific case
min_samples = 5  # Adjust based on your specific case
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
labels = dbscan.fit_predict(centroid)

# Add the cluster labels to the DataFrame
gradable_df['cluster'] = labels

# Plot the results
plt.imshow(original_img)
plt.axis('on')  # Ensure the axis is turned on

unique_labels = np.unique(labels)
for label in unique_labels:
    cluster_mask = gradable_df['cluster'] == label
    if label == -1:
        # Plot outliers in black color
        plt.scatter(gradable_df.loc[cluster_mask, 'centroid'].apply(lambda x: x[0]), gradable_df.loc[cluster_mask, 'centroid'].apply(lambda x: x[1]), color='black', label='Outliers')
    else:
        # Plot each cluster with a different color
        plt.scatter(gradable_df.loc[cluster_mask, 'centroid'].apply(lambda x: x[0]), gradable_df.loc[cluster_mask, 'centroid'].apply(lambda x: x[1]), label=f'Cluster {label}')

plt.legend()
plt.xlim(0, original_img.shape[1])
plt.ylim(original_img.shape[0], 0)  # Invert y-axis to match image coordinates

# Save the plot to the results directory
plot_path = os.path.join(output_directory, f"{img_path_name_only}_DBSCAN_clusters.png")
plt.savefig(plot_path)
plt.close()

# Save the updated DataFrame with cluster information
updated_csv_path = os.path.join(output_directory, f"{img_path_name_only}_gradable_clustered.csv")
gradable_df.to_csv(updated_csv_path, index=False)