"""
FinSight AI — SEC EDGAR MCP Tools
Direct REST API calls to SEC EDGAR (no API key required).
All requests include required User-Agent header per SEC guidelines.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import httpx
import pdfplumber
from io import BytesIO

from app.config import get_settings
from cache.disk_cache import TTL_LONG, TTL_MEDIUM, build_cache_key, cache_get, cache_set

logger = logging.getLogger(__name__)
settings = get_settings()

EDGAR_HEADERS = {
    "User-Agent": settings.EDGAR_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}
EDGAR_WWW_HEADERS = {
    "User-Agent": settings.EDGAR_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}

_RATE_LIMIT_SLEEP = 0.12  # SEC requires max ~10 req/sec


def _sleep() -> None:
    """Rate-limit sleep between EDGAR calls."""
    time.sleep(_RATE_LIMIT_SLEEP)


def search_company_cik(company_name: str) -> dict[str, str]:
    """
    Search SEC EDGAR for a company by name or ticker.
    Returns {ticker, name, cik} dict.
    """
    cache_key = build_cache_key("edgar_cik", company_name.upper())
    cached = cache_get(cache_key)
    if cached:
        return cached

    # Try company_tickers.json for exact ticker match first
    try:
        _sleep()
        with httpx.Client(headers=EDGAR_HEADERS, timeout=30.0) as client:
            resp = client.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers=EDGAR_WWW_HEADERS,
            )
            resp.raise_for_status()
            tickers_data = resp.json()

        query_upper = company_name.upper().strip()
        for _idx, entry in tickers_data.items():
            ticker = str(entry.get("ticker", "")).upper()
            title = str(entry.get("title", ""))
            cik_raw = str(entry.get("cik_str", "")).zfill(10)

            if ticker == query_upper or query_upper in title.upper():
                result = {
                    "ticker": ticker,
                    "name": title,
                    "cik": cik_raw,
                }
                cache_set(cache_key, result, ttl=TTL_LONG)
                return result

    except Exception as exc:
        logger.warning("EDGAR ticker search failed: %s", exc)

    # Fallback: EDGAR full-text search
    try:
        _sleep()
        with httpx.Client(headers=EDGAR_WWW_HEADERS, timeout=30.0) as client:
            resp = client.get(
                "https://efts.sec.gov/LATEST/search-index",
                params={"q": company_name, "dateRange": "custom", "forms": "10-K"},
                headers=EDGAR_WWW_HEADERS,
            )
            resp.raise_for_status()
            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])
            if hits:
                source = hits[0].get("_source", {})
                ticker = source.get("period_of_report", "")[:10]
                name = source.get("entity_name", company_name)
                cik = str(source.get("entity_id", "")).zfill(10)
                result = {"ticker": company_name.upper(), "name": name, "cik": cik}
                cache_set(cache_key, result, ttl=TTL_LONG)
                return result
    except Exception as exc:
        logger.warning("EDGAR full-text search failed: %s", exc)

    return {"ticker": company_name.upper(), "name": company_name, "cik": ""}


def get_recent_filings(
    cik: str,
    form_types: list[str] | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Get recent filings for a company from EDGAR submissions API.
    Returns list of filing dicts with accession_number, form_type, filing_date, etc.
    """
    if not cik:
        return []

    _form_types = form_types or ["10-K", "10-Q", "8-K"]
    cache_key = build_cache_key("edgar_filings", cik, ",".join(_form_types), str(limit))
    cached = cache_get(cache_key)
    if cached:
        return cached

    padded_cik = cik.zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{padded_cik}.json"

    try:
        _sleep()
        with httpx.Client(headers=EDGAR_HEADERS, timeout=30.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        periods = recent.get("reportDate", [])
        primary_docs = recent.get("primaryDocument", [])

        results: list[dict[str, Any]] = []
        for i, form in enumerate(forms):
            if form in _form_types and len(results) < limit:
                acc = accessions[i].replace("-", "") if i < len(accessions) else ""
                acc_formatted = accessions[i] if i < len(accessions) else ""
                primary_doc = primary_docs[i] if i < len(primary_docs) else ""

                # Build document URL
                acc_clean = acc_formatted.replace("-", "")
                doc_url = (
                    f"https://www.sec.gov/Archives/edgar/data/{int(cik)}"
                    f"/{acc_clean}/{primary_doc}"
                    if primary_doc
                    else ""
                )

                results.append({
                    "form_type": form,
                    "filing_date": dates[i] if i < len(dates) else "",
                    "period_of_report": periods[i] if i < len(periods) else "",
                    "accession_number": acc_formatted,
                    "document_url": doc_url,
                    "cik": cik,
                })

        cache_set(cache_key, results, ttl=TTL_MEDIUM)
        return results

    except Exception as exc:
        logger.error("Error fetching filings for CIK %s: %s", cik, exc)
        return []


def download_filing_text(cik: str, accession_number: str) -> str:
    """
    Download a filing and return cleaned plain text.
    Handles both HTML and PDF filings.
    Returns up to MAX_FILING_CHARS characters.
    """
    cache_key = build_cache_key("edgar_text", cik, accession_number)
    cached = cache_get(cache_key)
    if cached:
        return cached

    acc_clean = accession_number.replace("-", "")
    index_url = (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{acc_clean}-index.htm"
    )

    try:
        _sleep()
        with httpx.Client(headers=EDGAR_WWW_HEADERS, timeout=60.0, follow_redirects=True) as client:
            # Try to get the filing index first
            idx_resp = client.get(index_url)
            idx_text = idx_resp.text if idx_resp.status_code == 200 else ""

            # Find primary document from index
            doc_url = _find_primary_doc(idx_text, cik, acc_clean)

            if not doc_url:
                # Fallback: try common naming patterns
                doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{accession_number}.htm"

            _sleep()
            doc_resp = client.get(doc_url)
            doc_resp.raise_for_status()
            content_type = doc_resp.headers.get("content-type", "")

            if "pdf" in content_type.lower():
                text = _extract_pdf_text(doc_resp.content)
            else:
                text = _clean_html_text(doc_resp.text)

            # Limit to max chars
            text = text[:settings.MAX_FILING_CHARS]
            cache_set(cache_key, text, ttl=TTL_LONG)
            return text

    except Exception as exc:
        logger.error("Error downloading filing %s: %s", accession_number, exc)
        return ""


def _find_primary_doc(index_html: str, cik: str, acc_clean: str) -> str:
    """Extract primary document URL from EDGAR index HTML."""
    # Look for .htm or .txt document links
    patterns = [
        r'href="(/Archives/edgar/data/\d+/\d+/[^"]+\.htm)"',
        r'href="(/Archives/edgar/data/\d+/\d+/[^"]+\.txt)"',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, index_html, re.IGNORECASE)
        if matches:
            # Skip index files themselves
            for m in matches:
                if "index" not in m.lower():
                    return f"https://www.sec.gov{m}"
    return ""


def _clean_html_text(html: str) -> str:
    """Strip HTML tags and clean whitespace from filing text."""
    # Remove scripts and styles
    html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Decode common HTML entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&nbsp;", " ").replace("&#160;", " ")
    # Clean whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {3,}", " ", text)
    return text.strip()


def _extract_pdf_text(content: bytes) -> str:
    """Extract text from PDF bytes using pdfplumber."""
    try:
        text_parts: list[str] = []
        with pdfplumber.open(BytesIO(content)) as pdf:
            for page in pdf.pages[:50]:  # Limit to first 50 pages
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n\n".join(text_parts)
    except Exception as exc:
        logger.warning("PDF extraction error: %s", exc)
        return ""


def get_company_facts(cik: str) -> dict[str, Any]:
    """
    Fetch XBRL company facts from EDGAR (structured financial data).
    Returns raw facts dict from the companyfacts API.
    """
    cache_key = build_cache_key("edgar_facts", cik)
    cached = cache_get(cache_key)
    if cached:
        return cached

    padded_cik = cik.zfill(10)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{padded_cik}.json"

    try:
        _sleep()
        with httpx.Client(headers=EDGAR_HEADERS, timeout=60.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()

        # Extract key financial facts
        us_gaap = data.get("facts", {}).get("us-gaap", {})
        summary: dict[str, Any] = {
            "cik": cik,
            "entity_name": data.get("entityName", ""),
            "revenues": _extract_fact_values(us_gaap, "Revenues"),
            "net_income": _extract_fact_values(us_gaap, "NetIncomeLoss"),
            "total_assets": _extract_fact_values(us_gaap, "Assets"),
            "total_liabilities": _extract_fact_values(us_gaap, "Liabilities"),
            "eps": _extract_fact_values(us_gaap, "EarningsPerShareBasic"),
        }

        cache_set(cache_key, summary, ttl=TTL_MEDIUM)
        return summary

    except Exception as exc:
        logger.error("Error fetching company facts for CIK %s: %s", cik, exc)
        return {"cik": cik, "entity_name": "", "revenues": [], "net_income": []}


def _extract_fact_values(
    us_gaap: dict[str, Any], concept: str, max_entries: int = 8
) -> list[dict[str, Any]]:
    """Extract recent annual values for a given XBRL concept."""
    fact = us_gaap.get(concept, {})
    units = fact.get("units", {})
    usd_data = units.get("USD", units.get("pure", []))

    # Filter annual (10-K) entries and sort by end date
    annual = [
        entry for entry in usd_data
        if entry.get("form") in ("10-K", "10-K/A") and "end" in entry
    ]
    annual.sort(key=lambda x: x.get("end", ""), reverse=True)

    return [
        {"period": e.get("end", ""), "value": e.get("val", 0), "form": e.get("form", "")}
        for e in annual[:max_entries]
    ]
