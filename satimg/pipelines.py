"""Пайплайни обробки та реєстр окремих методів.

Пайплайн №1 «Текстурна сегментація земного покриву»:
    Кувахара (фільтрація) -> RGB->HSV (колір) -> GLCM (текстура) -> k-means (сегментація)
Пайплайн №2 «Виділення доріг і лінійних структур»:
    CLAHE (гістограма) -> Кенні (контури) -> закриття + чистка (морфологія) -> Хаф (прямі)

Кожна стадія повертає StageResult: дані для наступної стадії та
зображення для збереження проміжного результату.
"""
from dataclasses import dataclass, field

import numpy as np

from . import color, edges, filters, histogram, hough, morphology, segmentation, texture
from .core import box_sum, to_gray


@dataclass
class StageResult:
    """Результат однієї стадії пайплайна."""

    name: str
    category: str
    data: object
    preview: np.ndarray
    extra: dict = field(default_factory=dict)


def _hsv_features(hsv, glcm_maps, window):
    """Вектор ознак пікселя: колір (V, S cos H, S sin H, S) + 4 ознаки GLCM.

    Відтінок H — кут, тому кодується через cos/sin, щоб 359 і 1 градус
    вважались близькими. Колірні ознаки усереднюються в тому ж вікні, що
    й GLCM: інакше смугаста текстура (світлі й темні смуги) розпадається
    на два кластери за кольором. Яскравість V стоїть першою: за нею
    впорядковуються кластери, щоб кольори сегментів були стабільними.
    """
    hue = np.deg2rad(hsv[..., 0])
    sat, val = hsv[..., 1], hsv[..., 2]
    color_feats = np.stack([val, sat * np.cos(hue), sat * np.sin(hue), sat], axis=-1)
    color_feats = np.moveaxis(box_sum(np.moveaxis(color_feats, -1, 0), window), 0, -1)
    color_feats /= box_sum(np.ones(hsv.shape[:2]), window)[..., np.newaxis]
    return np.concatenate([color_feats, glcm_maps], axis=-1)


def run_pipeline1(img, params, seed=42):
    """Пайплайн №1. params — секція 'pipeline1' конфігурації."""
    kp, gp, mp = params["kuwahara"], params["glcm"], params["kmeans"]
    results = []

    filtered = filters.kuwahara(img, window=int(kp["window"]))
    results.append(StageResult("01_kuwahara", "фільтрація", filtered, filtered))

    hsv = color.rgb_to_hsv(filtered)
    results.append(StageResult("02_hsv", "колірні перетворення", hsv, color.hsv_to_display(hsv)))

    offsets = tuple(tuple(o) for o in gp.get("offsets", texture.DEFAULT_OFFSETS))
    maps = texture.glcm_feature_maps(hsv[..., 2] * 255.0, window=int(gp["window"]),
                                     levels=int(gp["levels"]), offsets=offsets)
    results.append(StageResult("03_glcm", "аналіз текстур", maps,
                               texture.features_to_display(maps)))

    features = _hsv_features(hsv, maps, int(gp["window"]))
    cw, tw = float(mp.get("color_weight", 1.0)), float(mp.get("texture_weight", 1.0))
    weights = [cw] * 4 + [tw] * maps.shape[-1]
    labels = segmentation.segment_kmeans(features, k=int(mp["k"]),
                                         max_iter=int(mp["max_iter"]),
                                         tol=float(mp["tol"]), seed=seed, weights=weights,
                                         n_init=int(mp.get("n_init", 3)))
    results.append(StageResult("04_kmeans", "сегментація", labels,
                               segmentation.labels_to_color(labels)))
    return results


def run_pipeline2(img, params, seed=42):  # pylint: disable=unused-argument
    """Пайплайн №2. params — секція 'pipeline2' конфігурації."""
    cp, kp, mp, hp = params["clahe"], params["canny"], params["morphology"], params["hough"]
    results = []

    gray = to_gray(img)
    enhanced = histogram.clahe(gray, tiles=tuple(cp["tiles"]),
                               clip_limit=float(cp["clip_limit"]))
    results.append(StageResult("01_clahe", "гістограмна корекція", enhanced, enhanced))

    edge_map = edges.canny(enhanced, sigma=float(kp["sigma"]),
                           low=float(kp["low"]), high=float(kp["high"]))
    results.append(StageResult("02_canny", "контурний аналіз", edge_map, edge_map))

    # Спершу прибираємо короткі фрагменти (текстура дахів, крон дерев),
    # потім закриттям з'єднуємо розриви вздовж довгих контурів.
    long_edges = morphology.remove_small_objects(edge_map, min_size=int(mp["min_size"]))
    cleaned = morphology.closing(long_edges, size=int(mp["closing_size"]))
    results.append(StageResult("03_morphology", "морфологічна обробка", cleaned, cleaned))

    # Голосують лише пікселі, що лежать на контурі Кенні, щоб закриття
    # не потовщувало лінії і не створювало зайвих голосів.
    voting = cleaned & edge_map
    lines, _ = hough.hough_lines(
        voting, theta_step=float(hp["theta_step"]), rho_step=float(hp["rho_step"]),
        threshold=float(hp["threshold"]),
        min_votes=int(round(float(hp["min_votes_frac"]) * min(img.shape[:2]))),
        num_peaks=int(hp["num_peaks"]), nms_rho=int(hp["nms_rho"]),
        nms_theta=int(hp["nms_theta"]), rho_window=int(hp.get("rho_window", 3)))
    overlay = hough.draw_lines(img, lines, color=(255, 0, 0), thickness=1)
    results.append(StageResult("04_hough", "перетворення Хафа", lines, overlay,
                               extra={"n_lines": len(lines)}))
    return results


PIPELINES = {"p1": run_pipeline1, "p2": run_pipeline2}

PIPELINE_TITLES = {
    "p1": "Текстурна сегментація земного покриву",
    "p2": "Виділення доріг і лінійних структур",
}


def _method_kuwahara(img, cfg, seed):  # pylint: disable=unused-argument
    return filters.kuwahara(img, int(cfg["pipeline1"]["kuwahara"]["window"]))


def _method_hsv(img, cfg, seed):  # pylint: disable=unused-argument
    return color.hsv_to_display(color.rgb_to_hsv(img))


def _method_glcm(img, cfg, seed):  # pylint: disable=unused-argument
    gp = cfg["pipeline1"]["glcm"]
    maps = texture.glcm_feature_maps(to_gray(img), int(gp["window"]), int(gp["levels"]),
                                     tuple(tuple(o) for o in gp["offsets"]))
    return texture.features_to_display(maps)


def _method_kmeans(img, cfg, seed):
    mp = cfg["pipeline1"]["kmeans"]
    hsv = color.rgb_to_hsv(img)
    labels = segmentation.segment_kmeans(hsv[..., [2, 1, 0]], int(mp["k"]),
                                         int(mp["max_iter"]), float(mp["tol"]), seed)
    return segmentation.labels_to_color(labels)


def _method_clahe(img, cfg, seed):  # pylint: disable=unused-argument
    cp = cfg["pipeline2"]["clahe"]
    return histogram.clahe(to_gray(img), tuple(cp["tiles"]), float(cp["clip_limit"]))


def _method_canny(img, cfg, seed):  # pylint: disable=unused-argument
    kp = cfg["pipeline2"]["canny"]
    return edges.canny(to_gray(img), float(kp["sigma"]), float(kp["low"]), float(kp["high"]))


def _method_morphology(img, cfg, seed):  # pylint: disable=unused-argument
    mp = cfg["pipeline2"]["morphology"]
    kp = cfg["pipeline2"]["canny"]
    edge_map = edges.canny(to_gray(img), float(kp["sigma"]), float(kp["low"]), float(kp["high"]))
    long_edges = morphology.remove_small_objects(edge_map, int(mp["min_size"]))
    return morphology.closing(long_edges, int(mp["closing_size"]))


def _method_hough(img, cfg, seed):
    return run_pipeline2(img, cfg["pipeline2"], seed)[-1].preview


METHODS = {
    "kuwahara": _method_kuwahara,
    "hsv": _method_hsv,
    "glcm": _method_glcm,
    "kmeans": _method_kmeans,
    "clahe": _method_clahe,
    "canny": _method_canny,
    "morphology": _method_morphology,
    "hough": _method_hough,
}
