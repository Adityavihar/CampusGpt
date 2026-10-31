import subprocess
import json
import re
import pytest
from pathlib import Path

INDEX_HTML_PATH = Path("templates/index.html")

def test_index_html_has_table_styles():
    content = INDEX_HTML_PATH.read_text(encoding="utf-8")
    assert ".table-wrapper" in content
    assert "table.chat-table" in content
    assert "table.chat-table th" in content
    assert "table.chat-table td" in content
    assert "preprocessMarkdown" in content
    assert "repairTableBlock" in content
    assert "renderMarkdownSafe" in content
    assert "formatMarkdown" in content


def test_markdown_rendering_all_seven_cases():
    """Extract JavaScript renderer directly from templates/index.html and test all 7 cases via Node.js."""
    html_content = INDEX_HTML_PATH.read_text(encoding="utf-8")
    
    # Extract script contents from index.html
    script_match = re.search(r"<script>([\s\S]*?)<\/script>", html_content)
    assert script_match is not None, "Could not find <script> in index.html"
    js_code = script_match.group(1)

    test_harness = """
const testCases = {
    caseA: `**Vinayaka Chaturthi Holiday**\\n- Date: 14 September 2026 (Monday)\\n- Status: No Class Day for both freshers and seniors.`,
    caseB: `| Student Group | Examination Type | Dates |\\n|---|---|---|\\n| Seniors | LAB FAT | Example dates |\\n| Freshers | Theory FAT | Example dates |`,
    caseC: `### Fall Semester 2026-27 Examination Schedule\\n\\nPlease find the dates below:\\n\\n| Student Group | Examination Type | Dates |\\n|---|---|---|\\n| Seniors | LAB FAT | Example dates |\\n| Freshers | Theory FAT | Example dates |\\n\\nContact the CoE for exam hall allocations.`,
    caseD1: `| Student Group | Examination Type | Dates |\\n| Seniors | LAB FAT | Example dates |\\n| Freshers | Theory FAT | Example dates |`,
    caseD2: `Student Group | Examination Type | Dates\\n---|---\\nSeniors | LAB FAT | Example dates\\nFreshers | Theory FAT`,
    caseE: `The Central Library is located in the Academic Block and remains open on all working days.`,
    caseF: `Security check: <script>alert("XSS")</script> and <img src=x onerror=alert(1)> and <a href="javascript:alert(2)">click me</a>.`
};

const results = {};
for (const [k, v] of Object.entries(testCases)) {
    results[k] = formatMarkdown(v);
}
console.log(JSON.stringify(results));
"""
    # Mock minimal DOM to avoid crashing in Node
    dom_mock = """
const document = {
    getElementById: () => ({ value: '', addEventListener: () => {}, style: {}, appendChild: () => {} })
};
"""
    full_node_code = dom_mock + "\n" + js_code + "\n" + test_harness
    
    result = subprocess.run(
        ["node", "-e", full_node_code],
        capture_output=True,
        text=True,
        check=True
    )
    
    res = json.loads(result.stdout)

    # Case A: Simple answer with bold text & bullets
    assert "<strong>Vinayaka Chaturthi Holiday</strong>" in res["caseA"]
    assert "<li>Date: 14 September 2026 (Monday)</li>" in res["caseA"]
    assert "<li>Status: No Class Day for both freshers and seniors.</li>" in res["caseA"]

    # Case B: Valid Markdown table
    assert '<div class="table-wrapper"><table class="chat-table">' in res["caseB"]
    assert ">Student Group</th>" in res["caseB"]
    assert ">Examination Type</th>" in res["caseB"]
    assert ">Dates</th>" in res["caseB"]
    assert ">Seniors</td>" in res["caseB"]
    assert ">LAB FAT</td>" in res["caseB"]
    assert ">Example dates</td>" in res["caseB"]

    # Case C: Heading, paragraphs & table
    assert "<h4>Fall Semester 2026-27 Examination Schedule</h4>" in res["caseC"]
    assert "<p>Please find the dates below:</p>" in res["caseC"]
    assert '<table class="chat-table">' in res["caseC"]
    assert "<p>Contact the CoE for exam hall allocations.</p>" in res["caseC"]

    # Case D1 & D2: Malformed tables (missing delimiter row & ragged columns)
    assert '<table class="chat-table">' in res["caseD1"]
    assert ">Student Group</th>" in res["caseD1"]
    assert ">Seniors</td>" in res["caseD1"]
    assert ">Freshers</td>" in res["caseD1"]

    assert '<table class="chat-table">' in res["caseD2"]
    assert ">Seniors</td>" in res["caseD2"]
    assert ">Freshers</td>" in res["caseD2"]

    # Case E: Plain text
    assert "<p>The Central Library is located in the Academic Block and remains open on all working days.</p>" in res["caseE"]

    # Case F: HTML/XSS Protection (No raw <script>, <img onerror>, or <a href> tags created)
    assert "&lt;script&gt;alert" in res["caseF"]
    assert "<script" not in res["caseF"]
    assert "<img" not in res["caseF"]
    assert "<a href=" not in res["caseF"]
