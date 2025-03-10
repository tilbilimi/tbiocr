import logging
import string
from collections import defaultdict
from typing import Any, List, Union

import cv2
import numpy as np
import torch
from doctr.io.elements import Document
from doctr.models import parseq
from doctr.models._utils import get_language
from doctr.models.detection.predictor import DetectionPredictor
from doctr.models.detection.zoo import detection_predictor
from doctr.models.predictor.base import _OCRPredictor
from doctr.models.recognition.predictor import RecognitionPredictor
from doctr.models.recognition.zoo import recognition_predictor
from doctr.utils.geometry import detach_scores
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from torch import nn

confidence_threshold = 0.75

reco_arch = "printed_v19.pt"
det_arch = "fast_base"

# Configure logging
afterword_symbols = "!?.,:;"
numbers = "0123456789"
other_symbols = string.punctuation + "«»…£€¥¢฿₸₽№°—"
space_symbol = " "
kazakh_letters = "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯЁабвгдежзийклмнопрстуфхцчшщъыьэюяёӘҒҚҢӨҰҮІҺәғқңөұүіһ"
english_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
all_letters = kazakh_letters + english_letters
all_symbols = numbers + other_symbols + space_symbol + all_letters


def get_ocr_predictor(
    det_arch: str = det_arch,
    reco_arch: str = reco_arch,
    pretrained=True,
    pretrained_backbone: bool = True,
    assume_straight_pages: bool = False,
    preserve_aspect_ratio: bool = True,
    symmetric_pad: bool = True,
    det_bs: int = 2,
    reco_bs: int = 128,
    detect_orientation: bool = False,
    straighten_pages: bool = False,
    detect_language: bool = False,
    bin_thresh: float = 0.3,
    box_thresh: float = 0.3,
):
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logging.info(f"Using device: {device}")

    device = torch.device(device)

    # Initialize predictor
    logging.info(f"Initializing predictor with device: {device}")
    reco_model = parseq(pretrained=False, pretrained_backbone=False, vocab=all_symbols)
    reco_model.to(device)
    reco_params = torch.load(f"./custom/{reco_arch}", map_location=device)
    reco_model.load_state_dict(reco_params)

    det_predictor = detection_predictor(
        det_arch,
        pretrained=pretrained,
        pretrained_backbone=pretrained_backbone,
        batch_size=det_bs,
        assume_straight_pages=assume_straight_pages,
        preserve_aspect_ratio=preserve_aspect_ratio,
        symmetric_pad=symmetric_pad,
    )

    # Recognition
    reco_predictor = recognition_predictor(
        reco_model,
        pretrained=pretrained,
        pretrained_backbone=pretrained_backbone,
        batch_size=reco_bs,
    )

    predictor = OCRPredictor(
        det_predictor,
        reco_predictor,
        assume_straight_pages=assume_straight_pages,
        preserve_aspect_ratio=preserve_aspect_ratio,
        symmetric_pad=symmetric_pad,
        detect_orientation=detect_orientation,
        straighten_pages=straighten_pages,
        detect_language=detect_language,
    )

    predictor.det_predictor.model.postprocessor.bin_thresh = bin_thresh
    predictor.det_predictor.model.postprocessor.box_thresh = box_thresh
    predictor.add_hook(CustomHook())

    return predictor


class OCRPredictor(nn.Module, _OCRPredictor):
    """Implements an object able to localize and identify text elements in a set of documents

    Args:
    ----
        det_predictor: detection module
        reco_predictor: recognition module
        assume_straight_pages: if True, speeds up the inference by assuming you only pass straight pages
            without rotated textual elements.
        straighten_pages: if True, estimates the page general orientation based on the median line orientation.
            Then, rotates page before passing it to the deep learning modules. The final predictions will be remapped
            accordingly. Doing so will improve performances for documents with page-uniform rotations.
        detect_orientation: if True, the estimated general page orientation will be added to the predictions for each
            page. Doing so will slightly deteriorate the overall latency.
        detect_language: if True, the language prediction will be added to the predictions for each
            page. Doing so will slightly deteriorate the overall latency.
        **kwargs: keyword args of `DocumentBuilder`
    """

    def __init__(
        self,
        det_predictor: DetectionPredictor,
        reco_predictor: RecognitionPredictor,
        assume_straight_pages: bool = True,
        straighten_pages: bool = False,
        preserve_aspect_ratio: bool = True,
        symmetric_pad: bool = True,
        detect_orientation: bool = False,
        detect_language: bool = False,
        **kwargs: Any,
    ) -> None:
        nn.Module.__init__(self)
        self.det_predictor = det_predictor.eval()  # type: ignore[attr-defined]
        self.reco_predictor = reco_predictor.eval()  # type: ignore[attr-defined]
        _OCRPredictor.__init__(
            self,
            assume_straight_pages,
            straighten_pages,
            preserve_aspect_ratio,
            symmetric_pad,
            detect_orientation,
            **kwargs,
        )
        self.detect_orientation = detect_orientation
        self.detect_language = detect_language

    @torch.inference_mode()
    def forward(
        self,
        pages: List[Union[np.ndarray, torch.Tensor]],
        **kwargs: Any,
    ) -> Document:
        # Dimension check
        if any(page.ndim != 3 for page in pages):
            raise ValueError(
                "incorrect input shape: all pages are expected to be multi-channel 2D images."
            )

        origin_page_shapes = [
            page.shape[:2] if isinstance(page, np.ndarray) else page.shape[-2:]
            for page in pages
        ]

        # Localize text elements
        loc_preds, out_maps = self.det_predictor(pages, return_maps=True, **kwargs)

        # Detect document rotation and rotate pages
        seg_maps = [
            np.where(
                out_map > getattr(self.det_predictor.model.postprocessor, "bin_thresh"),
                255,
                0,
            ).astype(np.uint8)
            for out_map in out_maps
        ]
        if self.detect_orientation:
            general_pages_orientations, origin_pages_orientations = self._get_orientations(pages, seg_maps)  # type: ignore[arg-type]
            orientations = [
                {"value": orientation_page, "confidence": None}
                for orientation_page in origin_pages_orientations
            ]
        else:
            orientations = None
            general_pages_orientations = None
            origin_pages_orientations = None
        if self.straighten_pages:
            pages = self._straighten_pages(pages, seg_maps, general_pages_orientations, origin_pages_orientations)  # type: ignore
            # Forward again to get predictions on straight pages
            loc_preds = self.det_predictor(pages, **kwargs)

        assert all(
            len(loc_pred) == 1 for loc_pred in loc_preds
        ), "Detection Model in ocr_predictor should output only one class"

        loc_preds = [list(loc_pred.values())[0] for loc_pred in loc_preds]
        # Detach objectness scores from loc_preds
        loc_preds, objectness_scores = detach_scores(loc_preds)
        # Check whether crop mode should be switched to channels first
        channels_last = len(pages) == 0 or isinstance(pages[0], np.ndarray)

        # Apply hooks to loc_preds if any
        for hook in self.hooks:
            loc_preds = hook(loc_preds)

        # Crop images
        crops, loc_preds = self._prepare_crops(
            pages,  # type: ignore[arg-type]
            loc_preds,
            channels_last=channels_last,
            assume_straight_pages=self.assume_straight_pages,
        )
        # Rectify crop orientation and get crop orientation predictions
        crop_orientations: Any = []
        # save crops to ./crops
        # os.makedirs("./crops", exist_ok=True)
        # for i, crop in enumerate(crops[0]):
        #     Image.fromarray(crop).save(f"./crops/{i}.png")

        # if not self.assume_straight_pages:
        # crops, loc_preds, _crop_orientations = self._rectify_crops(crops, loc_preds)
        #     crop_orientations = [
        #         {"value": orientation[0], "confidence": orientation[1]} for orientation in _crop_orientations
        #     ]

        # Identify character sequences
        word_preds = self.reco_predictor(
            [crop for page_crops in crops for crop in page_crops], **kwargs
        )
        if not crop_orientations:
            crop_orientations = [{"value": 0, "confidence": None} for _ in word_preds]

        boxes, text_preds, crop_orientations = self._process_predictions(
            loc_preds, word_preds, crop_orientations
        )

        if self.detect_language:
            languages = [
                get_language(" ".join([item[0] for item in text_pred]))
                for text_pred in text_preds
            ]
            languages_dict = [
                {"value": lang[0], "confidence": lang[1]} for lang in languages
            ]
        else:
            languages_dict = None

        out = self.doc_builder(
            pages,  # type: ignore[arg-type]
            boxes,
            objectness_scores,
            text_preds,
            origin_page_shapes,  # type: ignore[arg-type]
            crop_orientations,
            orientations,
            languages_dict,
        )
        return out


class CustomHook:
    def __call__(self, loc_preds):
        # Manipulate the location predictions here
        # 1. The outpout structure needs to be the same as the input location predictions
        # 2. Be aware that the coordinates are relative and needs to be between 0 and 1

        # return np.array([self.order_bbox_points(point) for loc_pred in loc_preds for point in loc_pred ])
        # iterate over each page and each box
        answer = []
        for page_idx, page_boxes in enumerate(loc_preds):
            bboxes = []
            for box_idx, box in enumerate(page_boxes):
                box = self.order_bbox_points(box)
                bboxes.append(box)
            answer.append(bboxes)
        return np.array(answer)

    def order_bbox_points(self, points):
        """
        Orders a list of four (x, y) points in the following order:
        top-left, top-right, bottom-right, bottom-left.

        Args:
            points (list of tuples): List of four (x, y) tuples.

        Returns:
            list of tuples: Ordered list of four (x, y) tuples.
        """
        if len(points) != 4:
            raise ValueError(
                "Exactly four points are required to define a quadrilateral."
            )

        # Convert points to NumPy array for easier manipulation
        pts = np.array(points)

        # Compute the sum and difference of the points
        sum_pts = pts.sum(axis=1)
        diff_pts = np.diff(pts, axis=1).flatten()

        # Initialize ordered points list
        ordered = [None] * 4

        # Top-Left point has the smallest sum
        ordered[0] = tuple(pts[np.argmin(sum_pts)])

        # Bottom-Right point has the largest sum
        ordered[2] = tuple(pts[np.argmax(sum_pts)])

        # Top-Right point has the smallest difference
        ordered[1] = tuple(pts[np.argmin(diff_pts)])

        # Bottom-Left point has the largest difference
        ordered[3] = tuple(pts[np.argmax(diff_pts)])

        return ordered


def geometry_to_coordinates(geometry, img_width, img_height):
    if len(geometry) == 2:
        (x0_rel, y0_rel), (x1_rel, y1_rel) = geometry
        x0 = int(x0_rel * img_width)
        y0 = int(y0_rel * img_height)
        x1 = int(x1_rel * img_width)
        y1 = int(y1_rel * img_height)
        # Bounding box with four corners
        all_four = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        return all_four
    else:
        # Bounding box with four corners
        all_four = [[int(x * img_width), int(y * img_height)] for x, y in geometry]
        return all_four


def page_to_coordinates(page_export):
    coordinates = []
    img_height, img_width = page_export["dimensions"]
    for block in page_export["blocks"]:
        for line in block["lines"]:
            for word in line["words"]:
                if (
                    word["confidence"] < confidence_threshold
                    and len(word["value"].strip()) > 1
                ):
                    logging.warning(
                        f"Skipping word with low confidence: {word['value']} confidence {word['confidence']}"
                    )
                    continue
                all_four = geometry_to_coordinates(
                    word["geometry"], img_width, img_height
                )
                coordinates.append((all_four, word["value"], word["confidence"]))

    return (coordinates, img_width, img_height)


def draw_boxes_with_labels(image, coordinates, font_path):
    """Бастапқы суретке шекаралар үстіне кішкентай белгілерді қою.

    Args:
        image: Бастапқы сурет (numpy массиві).
        out: predictor([image]) нәтижесі.
        font_path: TrueType қаріп файлының жолы.

    Returns:
        Шекаралар және белгілер қойылған сурет.
    """

    # Суретті PIL форматына түрлендіреміз
    img_with_boxes = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_with_boxes)
    draw = ImageDraw.Draw(img_pil)

    for coords, word, score in coordinates:
        # poligon
        coords = [(x, y) for x, y in coords]
        text_x, text_y = (
            min(coords, key=lambda x: x[0])[0],
            min(coords, key=lambda x: x[1])[1],
        )
        draw.polygon(coords, outline=(0, 255, 0, 125), width=1)
        font = ImageFont.truetype(font_path, 10)
        draw.text((text_x, max(text_y - 10, 0)), word, font=font, fill=(255, 0, 0))

    # Суретті қайтадан OpenCV форматына түрлендіреміз
    img_with_boxes = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # Суретті қайтарамыз
    return img_with_boxes


def generate_line_points(bbox, num_points=10):
    """
    Generates multiple points along the line connecting the left and right centers of a bounding box.

    Parameters:
    - bbox: List of four points [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
            in the order: TopLeft, TopRight, BottomRight, BottomLeft.
    - num_points: Number of points to generate along the line.

    Returns:
    - List of (x, y) tuples.
    """
    # Calculate left center (midpoint of TopLeft and BottomLeft)
    left_center_x = (bbox[0][0] + bbox[3][0]) / 2
    left_center_y = (bbox[0][1] + bbox[3][1]) / 2

    # Calculate right center (midpoint of TopRight and BottomRight)
    right_center_x = (bbox[1][0] + bbox[2][0]) / 2
    right_center_y = (bbox[1][1] + bbox[2][1]) / 2

    # Generate linearly spaced points between left center and right center
    x_values = np.linspace(left_center_x, right_center_x, num_points)
    y_values = np.linspace(left_center_y, right_center_y, num_points)

    points = list(zip(x_values, y_values))
    return points


def ocr_to_txt(coordinates):
    """
    Converts OCR output to a structured text file with lines using multiple points along connecting lines.
    Inserts empty lines when there's significant vertical spacing between lines.

    Parameters:
    - coordinates: List of tuples containing bounding box coordinates, word value, and score.
                   Each tuple is (([[x0, y0], [x1, y1], [x2, y2], [x3, y3]]), word, score)
    - img_width: Width of the image in pixels.
    - img_height: Height of the image in pixels.
    - output_file: Path to the output text file.
    """
    # Step 1: Compute multiple points for each word
    all_points = []
    words = []
    scaler = StandardScaler()
    points_per_word = 25  # Number of points to generate per word

    for bbox, word, score in coordinates:
        points = generate_line_points(bbox, num_points=points_per_word)
        all_points.extend(points)
        words.append(
            {
                "bbox": bbox,
                "word": word,
                "score": score,
                "points": points,  # Store the multiple points
            }
        )

    # Step 2: Scale the points
    scaled_points = scaler.fit_transform(all_points)
    scaled_points = [(c[0] / 5, c[1]) for c in scaled_points]
    scaled_points = np.array(scaled_points)

    # Step 3: Cluster points using DBSCAN
    # Parameters for DBSCAN can be tuned based on the specific OCR output
    # eps determines the maximum distance between two samples for them to be considered as in the same neighborhood
    # min_samples is set to the number of points per word to ensure entire words are clustered together
    db = DBSCAN(min_samples=2, eps=0.05).fit(scaled_points)  # eps might need adjustment
    labels = db.labels_

    # Map each point to its cluster label
    point_labels = labels.tolist()

    # Step 4: Assign words to clusters based on their points
    label_to_words = defaultdict(list)
    current_point = 0  # To keep track of which point belongs to which word

    for word in words:
        word_labels = point_labels[current_point : current_point + points_per_word]
        current_point += points_per_word

        # Count the frequency of each label in the word's points
        label_counts = defaultdict(int)
        for lbl in word_labels:
            label_counts[lbl] += 1

        # Assign the word to the most frequent label
        # If multiple labels have the same highest count, choose the smallest label (ignoring -1 for noise)
        if label_counts:
            # Exclude noise label (-1) when possible
            filtered_labels = {k: v for k, v in label_counts.items() if k != -1}
            if filtered_labels:
                assigned_label = max(filtered_labels, key=filtered_labels.get)
            else:
                assigned_label = -1  # Assign to noise
            label_to_words[assigned_label].append(word)

    # Remove noise cluster if present
    if -1 in label_to_words:
        print(
            f"Warning: {len(label_to_words[-1])} words assigned to noise cluster and will be ignored."
        )
        del label_to_words[-1]

    # Step 5: Sort words within each line
    sorted_lines = []
    line_heights = []  # To store heights of each line for median calculation
    line_y_bounds = []  # To store min and max y for each line

    for label, line_words in label_to_words.items():
        # Sort words based on their leftmost x-coordinate
        line_words_sorted = sorted(
            line_words, key=lambda w: min(point[0] for point in w["points"])
        )
        sorted_lines.append(line_words_sorted)

        # Compute y-bounds for the line
        y_values = []
        for word in line_words_sorted:
            y_coords = [point[1] for point in word["bbox"]]
            y_min = min(y_coords)
            y_max = max(y_coords)
            y_values.append([y_min, y_max])
        y_values = np.array(y_values)
        # Compute the median y-coordinates for the line by sorting only with y_min
        line_min_y_median = np.median(y_values[:, 0])
        line_max_y_median = np.median(y_values[:, 1])
        line_heights.append(line_max_y_median - line_min_y_median)
        line_y_bounds.append((line_min_y_median, line_max_y_median))

    # Step 6: Sort lines from top to bottom based on the average y-coordinate of their words
    sorted_lines, line_heights, line_y_bounds = zip(
        *sorted(
            zip(sorted_lines, line_heights, line_y_bounds),
            key=lambda item: np.median(
                [np.mean([p[1] for p in w["bbox"]]) for w in item[0]]
            ),
        )
    )

    sorted_lines = list(sorted_lines)
    line_heights = list(line_heights)
    line_y_bounds = list(line_y_bounds)

    # Step 8: Write sorted lines to the output text file with empty lines where necessary
    output_text = ""
    previous_line_median_y = None  # To track the max y of the previous line

    for idx, line in enumerate(sorted_lines):
        # Compute current line's min y
        current_line_min_y_median = line_y_bounds[idx][0]
        current_line_max_y_median = line_y_bounds[idx][1]
        current_line_median_height = line_heights[idx]
        current_line_median_y = (
            current_line_min_y_median + current_line_max_y_median
        ) / 2

        if previous_line_median_y is not None:
            # Compute vertical distance between lines
            vertical_distance = current_line_median_y - previous_line_median_y
            median_height = (
                current_line_median_height + previous_line_median_height
            ) / 2

            # If the vertical distance is greater than the median height, insert an empty line
            if vertical_distance > median_height * 2:
                output_text += "\n"  # Insert empty line

        # Write the current line's text
        line_text = " ".join([w["word"] for w in line])
        output_text += line_text + "\n"

        # Update the previous_line_max_y for the next iteration
        previous_line_median_y = current_line_median_y
        previous_line_median_height = current_line_median_height

    return output_text
