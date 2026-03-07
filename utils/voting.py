"""
utils/voting.py
===============
Majority-voting logic for video-level deepfake classification.

For a video, each sampled frame gets an individual prediction.
The final video verdict is determined by aggregating frame scores.

Strategies implemented:
  1. Hard majority vote  – count Real vs Fake frames
  2. Soft average vote   – average per-class probability scores
  3. Weighted vote       – weight frames by confidence
"""

import numpy as np
from typing import List, Tuple


# ─────────────────────────────────────────────
# Type alias: per-frame result
# ─────────────────────────────────────────────
# FrameResult = (predicted_class: int, real_prob: float, fake_prob: float)
FrameResult = Tuple[int, float, float]


# ─────────────────────────────────────────────
# 1. Hard Majority Vote
# ─────────────────────────────────────────────
def hard_majority_vote(frame_results: List[FrameResult]) -> dict:
    """
    Count frames predicted as Real (0) vs Fake (1).
    The class with more votes wins. Ties → Fake (cautious default).

    Returns:
        dict with keys: label, confidence, real_count, fake_count, total_frames
    """
    if not frame_results:
        return _empty_result("hard")

    real_count = sum(1 for cls, _, _ in frame_results if cls == 0)
    fake_count = len(frame_results) - real_count
    total = len(frame_results)

    if fake_count >= real_count:   # tie → fake (conservative)
        label = "Fake"
        confidence = fake_count / total
    else:
        label = "Real"
        confidence = real_count / total

    return {
        "strategy"    : "hard_majority",
        "label"       : label,
        "confidence"  : round(confidence * 100, 2),
        "real_count"  : real_count,
        "fake_count"  : fake_count,
        "total_frames": total,
    }


# ─────────────────────────────────────────────
# 2. Soft Average Vote (recommended)
# ─────────────────────────────────────────────
def soft_average_vote(frame_results: List[FrameResult]) -> dict:
    """
    Average the model's probability scores across all frames.
    More nuanced than hard voting – leverages model uncertainty.

    Returns:
        dict with label, confidence, avg_real_prob, avg_fake_prob
    """
    if not frame_results:
        return _empty_result("soft")

    real_probs = [r for _, r, _ in frame_results]
    fake_probs = [f for _, _, f in frame_results]

    avg_real = np.mean(real_probs)
    avg_fake = np.mean(fake_probs)

    if avg_fake >= avg_real:
        label = "Fake"
        confidence = avg_fake
    else:
        label = "Real"
        confidence = avg_real

    return {
        "strategy"     : "soft_average",
        "label"        : label,
        "confidence"   : round(float(confidence) * 100, 2),
        "avg_real_prob": round(float(avg_real) * 100, 2),
        "avg_fake_prob": round(float(avg_fake) * 100, 2),
        "total_frames" : len(frame_results),
    }


# ─────────────────────────────────────────────
# 3. Weighted Confidence Vote
# ─────────────────────────────────────────────
def weighted_confidence_vote(frame_results: List[FrameResult]) -> dict:
    """
    Weight each frame's vote by its prediction confidence.
    High-confidence frames have more influence.

    Weight = max(real_prob, fake_prob)   ← how certain the model was

    Returns:
        dict with label, confidence, weighted_score
    """
    if not frame_results:
        return _empty_result("weighted")

    weighted_fake = 0.0
    weighted_real = 0.0
    total_weight  = 0.0

    for cls, real_prob, fake_prob in frame_results:
        weight = max(real_prob, fake_prob)   # confidence in this frame
        if cls == 1:
            weighted_fake += weight * fake_prob
        else:
            weighted_real += weight * real_prob
        total_weight += weight

    if total_weight == 0:
        return _empty_result("weighted")

    score_real = weighted_real / total_weight
    score_fake = weighted_fake / total_weight

    if score_fake >= score_real:
        label = "Fake"
        confidence = score_fake / (score_real + score_fake + 1e-8)
    else:
        label = "Real"
        confidence = score_real / (score_real + score_fake + 1e-8)

    return {
        "strategy"      : "weighted_confidence",
        "label"         : label,
        "confidence"    : round(float(confidence) * 100, 2),
        "weighted_fake" : round(weighted_fake, 4),
        "weighted_real" : round(weighted_real, 4),
        "total_frames"  : len(frame_results),
    }


# ─────────────────────────────────────────────
# 4. Combined Ensemble
# ─────────────────────────────────────────────
def ensemble_vote(frame_results: List[FrameResult]) -> dict:
    """
    Run all three strategies and return the majority decision.
    Acts as a meta-ensemble.

    Returns the result from the strategy chosen by 2/3 or 3/3 vote,
    with the soft average confidence used as the final score.
    """
    hard   = hard_majority_vote(frame_results)
    soft   = soft_average_vote(frame_results)
    weight = weighted_confidence_vote(frame_results)

    votes = [hard["label"], soft["label"], weight["label"]]
    fake_votes = votes.count("Fake")
    real_votes = votes.count("Real")

    final_label = "Fake" if fake_votes >= real_votes else "Real"

    # Use soft average confidence as the official score
    final_confidence = soft["confidence"] if final_label == soft["label"] \
                       else 100 - soft["confidence"]

    return {
        "strategy"         : "ensemble",
        "label"            : final_label,
        "confidence"       : round(float(final_confidence), 2),
        "hard_label"       : hard["label"],
        "soft_label"       : soft["label"],
        "weighted_label"   : weight["label"],
        "total_frames"     : len(frame_results),
        "avg_fake_prob"    : soft.get("avg_fake_prob", 0),
        "avg_real_prob"    : soft.get("avg_real_prob", 0),
    }


# ─────────────────────────────────────────────
# 5. Risk Level Computation
# ─────────────────────────────────────────────
def compute_risk_level(label: str, confidence: float) -> str:
    """
    Translate prediction into a human-readable risk level.

    Risk categories:
      - Low    : Real, or Fake with low confidence (<60%)
      - Medium : Fake with moderate confidence (60–85%)
      - High   : Fake with high confidence (>85%)
    """
    if label == "Real":
        if confidence >= 90:
            return "Low"
        elif confidence >= 70:
            return "Low-Medium"
        else:
            return "Medium"   # uncertain real
    else:  # Fake
        if confidence >= 85:
            return "High"
        elif confidence >= 60:
            return "Medium"
        else:
            return "Low"   # uncertain fake


# ─────────────────────────────────────────────
# 6. Recommendation Generator
# ─────────────────────────────────────────────
def get_recommendation(label: str, risk_level: str) -> str:
    """Return a recommendation string for the forensic report."""
    if label == "Real" and "Low" in risk_level:
        return ("The media appears authentic. No immediate action required. "
                "Continue standard verification practices.")
    elif label == "Real":
        return ("The media is likely authentic but shows some uncertainty. "
                "Consider cross-verifying with original source.")
    elif risk_level == "Medium":
        return ("Possible deepfake detected. Do NOT share or act on this content. "
                "Seek expert verification before any decisions.")
    else:  # High / Fake
        return ("HIGH CONFIDENCE DEEPFAKE DETECTED. Immediately cease sharing "
                "this content. Report to the National Cyber Crime Portal "
                "(https://cybercrime.gov.in) and preserve evidence.")


# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────
def _empty_result(strategy: str) -> dict:
    return {
        "strategy"   : strategy,
        "label"      : "Unknown",
        "confidence" : 0.0,
        "total_frames": 0,
    }


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate 10 frame predictions: mostly fake
    mock_results: List[FrameResult] = [
        (1, 0.10, 0.90),
        (1, 0.15, 0.85),
        (0, 0.70, 0.30),
        (1, 0.05, 0.95),
        (1, 0.20, 0.80),
        (1, 0.08, 0.92),
        (0, 0.65, 0.35),
        (1, 0.12, 0.88),
        (1, 0.09, 0.91),
        (1, 0.18, 0.82),
    ]

    print("Hard  :", hard_majority_vote(mock_results))
    print("Soft  :", soft_average_vote(mock_results))
    print("Weight:", weighted_confidence_vote(mock_results))
    print("Ensem :", ensemble_vote(mock_results))

    result = ensemble_vote(mock_results)
    risk   = compute_risk_level(result["label"], result["confidence"])
    rec    = get_recommendation(result["label"], risk)
    print(f"\nRisk: {risk}")
    print(f"Recommendation: {rec}")
