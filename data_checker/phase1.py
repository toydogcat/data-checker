import re
import math
from collections import Counter
from typing import Dict, Any, List, Union, Optional
import pandas as pd
import numpy as np


class Phase1Checker:
    """
    階段一：統計與結構健康度 & 內容資訊豐富度 & 安全合規檢測
    Phase 1: Automated statistical health, information density, and safety checks.
    """

    # Common PII Regular Expressions
    PII_PATTERNS = {
        "taiwan_id": re.compile(r"\b[A-Za-z][1289]\d{8}\b"),
        "mobile_phone": re.compile(r"\b09\d{8}\b|\b0\d{1,2}-\d{6,8}\b"),
        "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
        "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
        "ip_address": re.compile(r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"),
    }

    # Toxic / Swear Words List (Bilingual demo set)
    DEFAULT_TOXIC_WORDS = [
        "幹", "三小", "靠北", "靠腰", "操你", "白痴", "智障", "蠢貨", "死全家",
        "fuck", "shit", "bitch", "bastard", "idiot", "asshole", "dumbass"
    ]

    def __init__(self, toxic_words: Optional[List[str]] = None):
        self.toxic_words = toxic_words or self.DEFAULT_TOXIC_WORDS
        self.toxic_regex = re.compile("|".join(re.escape(w) for w in self.toxic_words), re.IGNORECASE) if self.toxic_words else None

    @staticmethod
    def calculate_entropy(text: str) -> float:
        """Calculate Shannon entropy for a string (bits per char)."""
        if not text:
            return 0.0
        counts = Counter(text)
        total = len(text)
        return -sum((cnt / total) * math.log2(cnt / total) for cnt in counts.values())

    @staticmethod
    def calculate_ttr(texts: List[str]) -> float:
        """Calculate Type-Token Ratio (TTR) across a list of text strings."""
        tokens = []
        for t in texts:
            if not isinstance(t, str):
                continue
            # Simple word/character tokenizer
            words = re.findall(r"\w+", t.lower())
            tokens.extend(words)
        if not tokens:
            return 0.0
        unique_tokens = set(tokens)
        return len(unique_tokens) / len(tokens)

    def analyze_text_corpus(self, texts: List[str]) -> Dict[str, Any]:
        """
        Evaluate text corpus quality (Text Length, Entropy, TTR, PII, Toxicity).
        """
        clean_texts = [str(t) for t in texts if pd.notna(t)]
        total_count = len(clean_texts)
        if total_count == 0:
            return {"error": "Empty text dataset"}

        lengths = [len(t) for t in clean_texts]
        entropies = [self.calculate_entropy(t) for t in clean_texts]

        # Duplicate counts
        unique_texts = set(clean_texts)
        duplicate_count = total_count - len(unique_texts)
        duplication_rate = duplicate_count / total_count

        # PII counts
        pii_counts = {k: 0 for k in self.PII_PATTERNS}
        pii_matches_list = []

        # Toxicity counts
        toxic_matches_count = 0

        # Encoding issues detection (Replacement char \ufffd or non-printable ascii junk)
        encoding_issues_count = 0

        for idx, text in enumerate(clean_texts):
            # Encoding check
            if "\ufffd" in text or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text):
                encoding_issues_count += 1

            # PII scanner
            row_has_pii = False
            for pii_type, pattern in self.PII_PATTERNS.items():
                matches = pattern.findall(text)
                if matches:
                    pii_counts[pii_type] += len(matches)
                    row_has_pii = True
            if row_has_pii:
                pii_matches_list.append(idx)

            # Toxicity scanner
            if self.toxic_regex and self.toxic_regex.search(text):
                toxic_matches_count += 1

        ttr = self.calculate_ttr(clean_texts)

        return {
            "total_records": total_count,
            "unique_records": len(unique_texts),
            "duplicate_count": duplicate_count,
            "duplication_rate": round(duplication_rate * 100, 2),
            "length_metrics": {
                "min": int(np.min(lengths)),
                "max": int(np.max(lengths)),
                "avg": round(float(np.mean(lengths)), 2),
                "median": float(np.median(lengths))
            },
            "entropy_metrics": {
                "avg": round(float(np.mean(entropies)), 4),
                "min": round(float(np.min(entropies)), 4),
                "max": round(float(np.max(entropies)), 4)
            },
            "ttr": round(ttr, 4),
            "pii_summary": pii_counts,
            "total_pii_found": sum(pii_counts.values()),
            "toxic_records_count": toxic_matches_count,
            "toxicity_rate": round((toxic_matches_count / total_count) * 100, 2),
            "encoding_issues_count": encoding_issues_count,
        }

    def analyze_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate tabular pandas DataFrame (Missing rates, duplication, column types, PII).
        """
        total_rows, total_cols = df.shape
        if total_rows == 0:
            return {"error": "Empty DataFrame"}

        # Null / Sparsity analysis
        null_counts = df.isnull().sum().to_dict()
        empty_str_counts = {}
        for col in df.select_dtypes(include=["object", "string"]).columns:
            empty_str_counts[col] = int((df[col].astype(str).str.strip() == "").sum())

        missing_per_col = {}
        for col in df.columns:
            total_missing = int(null_counts.get(col, 0)) + int(empty_str_counts.get(col, 0))
            missing_per_col[col] = {
                "missing_count": total_missing,
                "missing_rate": round((total_missing / total_rows) * 100, 2)
            }

        overall_missing_cells = sum(item["missing_count"] for item in missing_per_col.values())
        overall_missing_rate = round((overall_missing_cells / (total_rows * total_cols)) * 100, 2)

        # Duplication analysis
        duplicated_rows = int(df.duplicated().sum())
        duplication_rate = round((duplicated_rows / total_rows) * 100, 2)

        # Column data types
        col_types = {col: str(dtype) for col, dtype in df.dtypes.items()}

        # Flatten string content to run text analysis
        text_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        combined_text_list = []
        for col in text_cols:
            combined_text_list.extend(df[col].dropna().astype(str).tolist())

        text_analysis = self.analyze_text_corpus(combined_text_list) if combined_text_list else {}

        # Compute overall quality score (0 to 100)
        # Score deductions: missing rate, duplication rate, PII occurrences, toxicity
        score = 100.0
        score -= min(30.0, overall_missing_rate * 0.8)
        score -= min(25.0, duplication_rate * 1.0)
        if text_analysis and "total_pii_found" in text_analysis:
            pii_penalty = min(20.0, text_analysis["total_pii_found"] * 2.0)
            score -= pii_penalty
        if text_analysis and "toxicity_rate" in text_analysis:
            toxic_penalty = min(15.0, text_analysis["toxicity_rate"] * 1.5)
            score -= toxic_penalty

        quality_score = max(0.0, round(score, 1))

        return {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "overall_missing_rate": overall_missing_rate,
            "duplication_rate": duplication_rate,
            "duplicated_rows": duplicated_rows,
            "column_missing_rates": missing_per_col,
            "column_types": col_types,
            "text_analysis": text_analysis,
            "quality_score": quality_score,
            "checklist_evaluation": {
                "completeness_pass": overall_missing_rate < 15.0,
                "uniqueness_pass": duplication_rate < 5.0,
                "safety_pass": text_analysis.get("total_pii_found", 0) == 0 and text_analysis.get("toxic_records_count", 0) == 0,
                "richness_pass": text_analysis.get("ttr", 0.0) > 0.3 if text_analysis else True
            }
        }
