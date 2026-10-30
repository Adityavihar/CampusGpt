import re
import math
import logging
from pathlib import Path
from typing import Optional, Union
from PIL import Image

logger = logging.getLogger(__name__)


def calculate_attendance_recovery(attended: int, total: int, target: float = 75.0) -> dict:
    """
    Calculate classes needed to reach target percentage,
    or margin of classes that can be safely missed.
    """
    if total <= 0:
        return {"classes_needed": 0, "classes_can_miss": 0}

    current_pct = (attended / total) * 100

    if current_pct < target:
        # (attended + x) / (total + x) >= target / 100
        # 100 * attended + 100 * x >= target * total + target * x
        # x * (100 - target) >= target * total - 100 * attended
        t_dec = target / 100.0
        numerator = (t_dec * total) - attended
        denominator = 1.0 - t_dec
        needed = math.ceil(numerator / denominator) if denominator > 0 else 0
        return {
            "classes_needed": max(0, needed),
            "classes_can_miss": 0
        }
    else:
        # attended / (total + y) >= target / 100
        # total + y <= attended / (target / 100)
        t_dec = target / 100.0
        max_total = attended / t_dec
        can_miss = math.floor(max_total - total)
        return {
            "classes_needed": 0,
            "classes_can_miss": max(0, can_miss)
        }


def parse_attendance_text(raw_text: str) -> dict:
    """
    Parse subjects and attendance values from text.
    Handles varied formatting (tabular, key-value, ratios).
    """
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    records = []
    
    # Common subject patterns:
    # 1. "Computer Networks: 32 / 40" or "CSE1001 Data Structures 28/35 80%"
    # 2. "Maths - Attended: 25 Total: 30"
    ratio_pattern = re.compile(
        r"^(?P<subject>[A-Za-z0-9\s,&.\-_()]+?)\s*[:|\-]?\s+(?P<attended>\d{1,3})\s*(?:/|of|out of)\s*(?P<total>\d{1,3})",
        re.IGNORECASE
    )
    keyword_pattern = re.compile(
        r"^(?P<subject>[A-Za-z0-9\s,&.\-_()]+?)\s*[:|\-]?\s*attended\s*[:=]?\s*(?P<attended>\d{1,3}).*?total\s*[:=]?\s*(?P<total>\d{1,3})",
        re.IGNORECASE
    )
    tabular_pattern = re.compile(
        r"^(?P<subject>[A-Za-z][A-Za-z0-9\s,&.\-_()]{2,40})\s+(?P<attended>\d{1,3})\s+(?P<total>\d{1,3})(?:\s+(?P<pct>\d{1,3}(?:\.\d+)?))?",
        re.IGNORECASE
    )

    seen_subjects = set()

    for line in lines:
        # Skip header lines
        if any(h in line.lower() for h in ["attendance report", "subject code", "course title", "s.no", "serial no"]):
            continue

        match = (
            keyword_pattern.search(line)
            or ratio_pattern.search(line)
            or tabular_pattern.search(line)
        )

        if match:
            sub = match.group("subject").strip(" :-|")
            # Filter out false positives like "Total Classes", "Semester 1"
            if len(sub) < 3 or sub.lower() in ["total", "status", "percentage", "present", "absent"]:
                continue

            try:
                att = int(match.group("attended"))
                tot = int(match.group("total"))
            except (ValueError, TypeError):
                continue

            # Basic validation
            if tot > 0 and att <= tot:
                sub_key = sub.lower()
                if sub_key not in seen_subjects:
                    seen_subjects.add(sub_key)
                    pct = round((att / tot) * 100, 2)
                    recovery = calculate_attendance_recovery(att, tot, target=75.0)

                    records.append({
                        "subject": sub,
                        "attended": att,
                        "total": tot,
                        "percentage": pct,
                        "is_shortage": pct < 75.0,
                        "status": "Safe" if pct >= 75.0 else "Shortage Warning (<75%)",
                        "classes_needed_for_75": recovery["classes_needed"],
                        "classes_can_miss": recovery["classes_can_miss"]
                    })

    # Calculate overall metrics
    total_attended = sum(r["attended"] for r in records)
    total_classes = sum(r["total"] for r in records)
    overall_pct = round((total_attended / total_classes) * 100, 2) if total_classes > 0 else 0.0

    shortage_count = sum(1 for r in records if r["is_shortage"])

    return {
        "success": True,
        "records_count": len(records),
        "overall": {
            "total_attended": total_attended,
            "total_conducted": total_classes,
            "overall_percentage": overall_pct,
            "has_shortage": shortage_count > 0,
            "shortage_courses_count": shortage_count
        },
        "subjects": records,
        "disclaimer": "Calculations are based on extracted document text or OCR. Please cross-check with the official VTOP student portal."
    }


def process_attendance_file(file_path: Union[str, Path]) -> dict:
    """
    Process an uploaded attendance document (PDF or image).
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return {"success": False, "error": f"File not found: {file_path.name}"}

    suffix = file_path.suffix.lower()
    extracted_text = ""

    try:
        if suffix in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"]:
            from services.ocr_service import ocr_image
            with Image.open(file_path) as img:
                extracted_text = ocr_image(img)
        elif suffix == ".pdf":
            from services.document_parser import extract_text_from_pdf
            pages = extract_text_from_pdf(str(file_path))
            extracted_text = "\n".join(p["text"] for p in pages)
        elif suffix in [".txt", ".csv"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                extracted_text = f.read()
        else:
            return {"success": False, "error": f"Unsupported file type: {suffix}"}

        if not extracted_text.strip():
            return {
                "success": False,
                "error": "No readable text could be extracted from the document."
            }

        result = parse_attendance_text(extracted_text)
        result["raw_text_preview"] = extracted_text[:300]
        return result

    except Exception as e:
        logger.exception(f"Error processing attendance file {file_path.name}")
        return {"success": False, "error": f"Attendance processing failed: {str(e)}"}
