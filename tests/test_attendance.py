import pytest
from services.attendance_service import (
    calculate_attendance_recovery,
    parse_attendance_text
)


def test_calculate_attendance_recovery():
    # 22 attended out of 32 conducted = 68.75% (< 75%)
    # To reach 75%: (22 + 8) / (32 + 8) = 30 / 40 = 75%
    rec = calculate_attendance_recovery(22, 32, target=75.0)
    assert rec["classes_needed"] == 8
    assert rec["classes_can_miss"] == 0

    # 28 attended out of 35 conducted = 80% (>= 75%)
    # Can miss 2: 28 / 37 = 75.68%
    rec_safe = calculate_attendance_recovery(28, 35, target=75.0)
    assert rec_safe["classes_needed"] == 0
    assert rec_safe["classes_can_miss"] == 2


def test_parse_attendance_text():
    sample = """
    Data Structures: 28 / 35
    Operating Systems 22 32
    Computer Networks - Attended: 30 Total: 40
    """
    res = parse_attendance_text(sample)
    assert res["success"] is True
    assert res["records_count"] == 3
    assert res["overall"]["total_attended"] == 80
    assert res["overall"]["total_conducted"] == 107
    assert res["overall"]["has_shortage"] is True

    subjects = {s["subject"]: s for s in res["subjects"]}
    assert "Operating Systems" in subjects
    assert subjects["Operating Systems"]["is_shortage"] is True
    assert subjects["Operating Systems"]["classes_needed_for_75"] == 8
