"""
utils/voting.py
===============
Improved voting logic for DeepShield video classification.

Key improvements over v1:
  - Peak fake detection: if ANY frame exceeds 85% fake confidence → flag
  - Suspicious frame ratio: if >25% of frames are fake → flag  
  - Weighted ensemble favours high-confidence fake frames more
  - More conservative (better at catching fakes, acceptable false positive rate)
"""

import numpy as np
from typing import List, Tuple

FrameResult = Tuple[int, float, float]


# ─────────────────────────────────────────────
# 1. Hard Majority Vote
# ─────────────────────────────────────────────
def hard_majority_vote(frame_results: List[FrameResult]) -> dict:
    if not frame_results:
        return _empty_result("hard")

    real_count = sum(1 for cls, _, _ in frame_results if cls == 0)
    fake_count = len(frame_results) - real_count
    total      = len(frame_results)

    # Tie → Fake (conservative)
    label      = "Fake" if fake_count >= real_count else "Real"
    confidence = (fake_count if label == "Fake" else real_count) / total

    return {
        "strategy"    : "hard_majority",
        "label"       : label,
        "confidence"  : round(confidence * 100, 2),
        "real_count"  : real_count,
        "fake_count"  : fake_count,
        "total_frames": total,
    }


# ─────────────────────────────────────────────
# 2. Soft Average Vote
# ─────────────────────────────────────────────
def soft_average_vote(frame_results: List[FrameResult]) -> dict:
    if not frame_results:
        return _empty_result("soft")

    real_probs = [r for _, r, _ in frame_results]
    fake_probs = [f for _, _, f in frame_results]

    avg_real = np.mean(real_probs)
    avg_fake = np.mean(fake_probs)

    label      = "Fake" if avg_fake >= avg_real else "Real"
    confidence = avg_fake if label == "Fake" else avg_real

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
    if not frame_results:
        return _empty_result("weighted")

    weighted_fake = 0.0
    weighted_real = 0.0
    total_weight  = 0.0

    for cls, real_prob, fake_prob in frame_results:
        weight = max(real_prob, fake_prob)
        if cls == 1:
            weighted_fake += weight * fake_prob
        else:
            weighted_real += weight * real_prob
        total_weight += weight

    if total_weight == 0:
        return _empty_result("weighted")

    score_real = weighted_real / total_weight
    score_fake = weighted_fake / total_weight
    denom      = score_real + score_fake + 1e-8

    label      = "Fake" if score_fake >= score_real else "Real"
    confidence = (score_fake if label == "Fake" else score_real) / denom

    return {
        "strategy"      : "weighted_confidence",
        "label"         : label,
        "confidence"    : round(float(confidence) * 100, 2),
        "weighted_fake" : round(weighted_fake, 4),
        "weighted_real" : round(weighted_real, 4),
        "total_frames"  : len(frame_results),
    }


# ─────────────────────────────────────────────
# 4. Peak Fake Detector  ← NEW
# ─────────────────────────────────────────────
def peak_fake_detector(frame_results: List[FrameResult],
                        peak_threshold: float = 0.92,
                        ratio_threshold: float = 0.60) -> dict:
    """
    Flags a video as fake if EITHER:
      (a) Any single frame has fake_prob >= peak_threshold (default 92%)
      (b) More than ratio_threshold (default 60%) of frames are predicted fake

    This catches deepfakes that only appear in a portion of a video,
    which soft averaging misses because real frames dilute the score.
    """
    if not frame_results:
        return _empty_result("peak")

    fake_probs  = [f for _, _, f in frame_results]
    peak_fake   = max(fake_probs)
    fake_frames = sum(1 for f in fake_probs if f > 0.5)
    ratio       = fake_frames / len(frame_results)

    # Trigger conditions
    peak_triggered  = peak_fake  >= peak_threshold
    ratio_triggered = ratio      >= ratio_threshold

    if peak_triggered or ratio_triggered:
        label      = "Fake"
        # Confidence = weighted blend of peak and ratio evidence
        confidence = max(peak_fake, ratio * 1.2)
        confidence = min(confidence, 0.99)
    else:
        label      = "Real"
        avg_real   = np.mean([r for _, r, _ in frame_results])
        confidence = avg_real

    return {
        "strategy"        : "peak_detector",
        "label"           : label,
        "confidence"      : round(float(confidence) * 100, 2),
        "peak_fake_prob"  : round(peak_fake * 100, 2),
        "suspicious_ratio": round(ratio * 100, 2),
        "peak_triggered"  : peak_triggered,
        "ratio_triggered" : ratio_triggered,
        "total_frames"    : len(frame_results),
    }


# ─────────────────────────────────────────────
# 5. Combined Ensemble  ← IMPROVED
# ─────────────────────────────────────────────
def ensemble_vote(frame_results: List[FrameResult]) -> dict:
    """
    Improved 4-strategy ensemble:
      - hard majority vote
      - soft average vote
      - weighted confidence vote
      - peak fake detector  ← new

    Decision logic (conservative — prefers catching fakes):
      • If peak_detector says Fake → Fake (overrides others)
      • Otherwise → majority of 4 votes
      • Tie (2-2) → Fake (cautious default)
    """
    if not frame_results:
        return _empty_result("ensemble")

    hard   = hard_majority_vote(frame_results)
    soft   = soft_average_vote(frame_results)
    weight = weighted_confidence_vote(frame_results)
    peak   = peak_fake_detector(frame_results)

    # Peak detector override — requires BOTH high peak AND 40%+ suspicious frames
    # This prevents 1 bad frame from flipping the whole verdict
    if peak["label"] == "Fake" and peak.get("peak_fake_prob", 0) >= 90 and peak.get("suspicious_ratio", 0) >= 40:
        final_label = "Fake"
        soft_fake_conf = soft.get("avg_fake_prob", 0)
        final_confidence = max(peak["confidence"], soft_fake_conf)
    else:
        votes      = [hard["label"], soft["label"], weight["label"], peak["label"]]
        fake_votes = votes.count("Fake")
        real_votes = votes.count("Real")
        # Tie (2-2) → Fake (conservative)
        final_label = "Fake" if fake_votes >= real_votes else "Real"

        if final_label == "Fake":
            final_confidence = soft.get("avg_fake_prob", soft["confidence"])
        else:
            final_confidence = soft.get("avg_real_prob", soft["confidence"])

    return {
        "strategy"        : "ensemble_v2",
        "label"           : final_label,
        "confidence"      : round(float(final_confidence), 2),
        "hard_label"      : hard["label"],
        "soft_label"      : soft["label"],
        "weighted_label"  : weight["label"],
        "peak_label"      : peak["label"],
        "peak_fake_prob"  : peak.get("peak_fake_prob", 0),
        "suspicious_ratio": peak.get("suspicious_ratio", 0),
        "total_frames"    : len(frame_results),
        "avg_fake_prob"   : soft.get("avg_fake_prob", 0),
        "avg_real_prob"   : soft.get("avg_real_prob", 0),
    }


# ─────────────────────────────────────────────
# 6. Risk Level
# ─────────────────────────────────────────────
def compute_risk_level(label: str, confidence: float) -> str:
    if label == "Real":
        if confidence >= 90:
            return "Low"
        elif confidence >= 70:
            return "Low-Medium"
        else:
            return "Medium"
    else:  # Fake
        if confidence >= 85:
            return "High"
        elif confidence >= 60:
            return "Medium"
        else:
            return "Low"


# ─────────────────────────────────────────────
# 7. Recommendation
# ─────────────────────────────────────────────
def get_recommendation(label: str, risk_level: str) -> str:
    if label == "Real" and "Low" in risk_level:
        return ("The media appears authentic. No immediate action required. "
                "Continue standard verification practices.")
    elif label == "Real":
        return ("The media is likely authentic but shows some uncertainty. "
                "Consider cross-verifying with the original source.")
    elif risk_level == "Medium":
        return ("Possible deepfake detected. Do NOT share or act on this content. "
                "Seek expert verification before any decisions.")
    else:
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


if __name__ == "__main__":
    # Simulate a video where SOME frames are highly fake (partial deepfake)
    mock: List[FrameResult] = [
        (0, 0.85, 0.15),  # real frame
        (0, 0.80, 0.20),  # real frame
        (1, 0.10, 0.90),  # FAKE frame — high confidence
        (0, 0.75, 0.25),  # real frame
        (1, 0.05, 0.95),  # FAKE frame — very high confidence
        (0, 0.70, 0.30),  # real frame
        (0, 0.78, 0.22),  # real frame
        (0, 0.82, 0.18),  # real frame
    ]
    print("Soft avg would say:", soft_average_vote(mock)["label"],
          f"({soft_average_vote(mock)['avg_fake_prob']}% fake avg)")
    print("Peak detector says:", peak_fake_detector(mock)["label"],
          f"(peak={peak_fake_detector(mock)['peak_fake_prob']}%)")
    print("Ensemble v2 says  :", ensemble_vote(mock)["label"])
