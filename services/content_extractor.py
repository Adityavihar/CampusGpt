import os
import re
import io
import socket
import ipaddress
import logging
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
from html.parser import HTMLParser
from typing import Optional, Tuple
import requests

logger = logging.getLogger(__name__)

# Maximum content size allowed for downloaded webpages (5 MB)
MAX_WEBPAGE_BYTES = 5 * 1024 * 1024
REQUEST_TIMEOUT = 10  # seconds


# =========================================================
# SSRF PROTECTION & URL VALIDATOR
# =========================================================

def is_safe_url(url: str) -> Tuple[bool, str]:
    """
    Validate that a URL is safe to fetch and protects against SSRF:
    - Must use http or https scheme.
    - Hostname must resolve to a valid public IP.
    - Loopback, private, link-local, multicast, and reserved addresses are strictly blocked.
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception as e:
        return False, f"Invalid URL structure: {e}"

    if parsed.scheme not in ("http", "https"):
        return False, f"Unsupported URL scheme '{parsed.scheme}'. Only http and https are permitted."

    hostname = parsed.hostname
    if not hostname:
        return False, "URL is missing a valid hostname."

    # Disallow known localhost aliases
    if hostname.lower() in ("localhost", "127.0.0.1", "0.0.0.0", "::1", "local"):
        return False, "Access to localhost or loopback destinations is forbidden."

    # Resolve hostname to IP addresses
    try:
        addr_info = socket.getaddrinfo(hostname, None)
    except socket.gaierror as e:
        return False, f"Could not resolve hostname '{hostname}': {e}"

    for family, _, _, _, sockaddr in addr_info:
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            return False, f"Invalid resolved IP address: {ip_str}"

        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return False, f"Access to private/internal network destination ({ip_str}) is forbidden."

    return True, "URL is valid and safe."


# =========================================================
# HTML TEXT EXTRACTOR (BUILT-IN HTMLPARSER)
# =========================================================

class CleanHTMLParser(HTMLParser):
    """
    Strips scripts, styles, navigations, footers, and extract structured main text.
    """
    IGNORE_TAGS = {
        "script", "style", "noscript", "svg", "canvas", "header",
        "footer", "nav", "aside", "form", "button", "iframe"
    }

    BLOCK_TAGS = {
        "p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "div",
        "section", "article", "tr", "table", "blockquote", "hr"
    }

    def __init__(self):
        super().__init__()
        self.ignore_depth = 0
        self.text_parts = []
        self.title_parts = []
        self.in_title = False
        self.meta_description = ""

    def handle_starttag(self, tag, attrs):
        tag_lower = tag.lower()
        if tag_lower in self.IGNORE_TAGS:
            self.ignore_depth += 1
        elif tag_lower == "title":
            self.in_title = True
        elif tag_lower == "meta":
            attr_dict = {k.lower(): v for k, v in attrs}
            if attr_dict.get("name", "").lower() in ("description", "og:description"):
                self.meta_description = attr_dict.get("content", "")

        if tag_lower in self.BLOCK_TAGS and self.ignore_depth == 0:
            self.text_parts.append("\n")

    def handle_endtag(self, tag):
        tag_lower = tag.lower()
        if tag_lower in self.IGNORE_TAGS:
            if self.ignore_depth > 0:
                self.ignore_depth -= 1
        elif tag_lower == "title":
            self.in_title = False

        if tag_lower in self.BLOCK_TAGS and self.ignore_depth == 0:
            self.text_parts.append("\n")

    def handle_data(self, data):
        if self.in_title:
            self.title_parts.append(data)
        elif self.ignore_depth == 0:
            text = data.strip()
            if text:
                self.text_parts.append(text + " ")

    def get_clean_text(self) -> str:
        raw = "".join(self.text_parts)
        # Normalize multiple spaces and blank lines
        clean = re.sub(r"[ \t]+", " ", raw)
        clean = re.sub(r"\n\s*\n+", "\n\n", clean)
        return clean.strip()

    get_text = get_clean_text

    def get_title(self) -> str:
        return " ".join(self.title_parts).strip()


def fetch_and_extract_webpage(url: str) -> dict:
    """
    Safely fetch a webpage and extract clean content and title.
    Enforces SSRF validation, redirects check, timeout, and response size limits.
    """
    safe, msg = is_safe_url(url)
    if not safe:
        raise ValueError(f"SSRF Protection blocked URL: {msg}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CampusGPT/2.0 Academic Assistant",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }

    # Fetch with stream=True to prevent reading unbounded payload
    session = requests.Session()
    session.max_redirects = 5

    try:
        response = session.get(
            url,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
            stream=True,
            allow_redirects=True
        )
        
        # Verify final redirected URL is also safe
        final_safe, final_msg = is_safe_url(response.url)
        if not final_safe:
            raise ValueError(f"Redirected to unsafe URL ({response.url}): {final_msg}")

        response.raise_for_status()

        # Check content type
        content_type = response.headers.get("Content-Type", "").lower()
        if not any(t in content_type for t in ("text/html", "application/xhtml+xml", "text/plain")):
            raise ValueError(f"Unsupported content type '{content_type}'. Expected text/html.")

        # Read safely up to MAX_WEBPAGE_BYTES
        content = bytearray()
        for chunk in response.iter_content(chunk_size=16384):
            content.extend(chunk)
            if len(content) > MAX_WEBPAGE_BYTES:
                raise ValueError(f"Webpage content exceeds maximum allowed size of {MAX_WEBPAGE_BYTES // (1024*1024)} MB.")

        encoding = response.encoding or "utf-8"
        try:
            html_text = content.decode(encoding, errors="replace")
        except Exception:
            html_text = content.decode("utf-8", errors="replace")

        parser = CleanHTMLParser()
        parser.feed(html_text)

        title = parser.get_title() or urllib.parse.urlparse(url).path.split("/")[-1] or "Webpage Information"
        body_text = parser.get_clean_text()

        if not body_text:
            raise ValueError("No readable textual content could be extracted from this webpage.")

        return {
            "title": title,
            "url": response.url,
            "text": body_text,
            "description": parser.meta_description,
            "status_code": response.status_code
        }

    except requests.exceptions.Timeout:
        raise TimeoutError(f"Connection to '{url}' timed out after {REQUEST_TIMEOUT}s.")
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to fetch webpage '{url}': {e}")


# =========================================================
# DOCX & TEXT FILE EXTRACTORS
# =========================================================

def extract_text_from_txt(file_path_or_bytes) -> str:
    """
    Extract text from a .txt or .md file with encoding fallbacks.
    """
    if isinstance(file_path_or_bytes, (str, os.PathLike)):
        with open(file_path_or_bytes, "rb") as f:
            raw_bytes = f.read()
    else:
        raw_bytes = file_path_or_bytes

    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return raw_bytes.decode(enc).strip()
        except UnicodeDecodeError:
            continue

    return raw_bytes.decode("utf-8", errors="replace").strip()


def extract_text_from_docx(file_path_or_bytes) -> str:
    """
    Extract text paragraphs and tables from a DOCX file using standard library zipfile & XML.
    No extra dependencies required.
    """
    if isinstance(file_path_or_bytes, (str, os.PathLike)):
        with open(file_path_or_bytes, "rb") as f:
            raw_bytes = f.read()
    else:
        raw_bytes = file_path_or_bytes

    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as z:
            xml_content = z.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            namespaces = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            
            paragraphs = []
            for p in tree.iterfind(".//w:p", namespaces):
                texts = [node.text for node in p.iterfind(".//w:t", namespaces) if node.text]
                if texts:
                    paragraphs.append("".join(texts).strip())
            
            return "\n\n".join([p for p in paragraphs if p]).strip()
    except Exception as e:
        logger.error(f"Failed to parse DOCX: {e}")
        raise ValueError(f"Could not extract text from DOCX document: {e}")
