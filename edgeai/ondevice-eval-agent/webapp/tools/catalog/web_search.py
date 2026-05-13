"""
Web Search Tool

Allows the agent to search the web for information about ML models,
frameworks, and related topics to provide better context.
"""

import ipaddress
import logging
import re
import socket
from typing import Dict, Any, List
from urllib.parse import quote_plus, urlparse

from tools.base import ok, error_response
from tools.registry import register_tool

logger = logging.getLogger(__name__)

# Check for available HTTP libraries
REQUESTS_AVAILABLE = False
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    logger.warning("requests library not available for web search")


def _extract_text_from_html(html: str, max_length: int = 2000) -> str:
    """Extract readable text from HTML content."""
    # Remove script and style elements
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', html)
    
    # Decode HTML entities
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&quot;', '"', text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Truncate to max length
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    return text


def _search_duckduckgo(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Search using DuckDuckGo HTML (no API key required).
    Returns list of search results with title, url, and snippet.
    """
    if not REQUESTS_AVAILABLE:
        return []
    
    try:
        # DuckDuckGo HTML search
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; EdgeAI-Agent/1.0)'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        html = response.text
        results = []
        
        # Parse results from HTML
        # DuckDuckGo HTML format: <a class="result__a" href="...">title</a>
        # <a class="result__snippet">snippet</a>
        
        # Find result blocks
        result_pattern = r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>([^<]*)</a>'
        snippet_pattern = r'<a[^>]*class="result__snippet"[^>]*>([^<]*)</a>'
        
        titles_urls = re.findall(result_pattern, html)
        snippets = re.findall(snippet_pattern, html)
        
        for i, (url, title) in enumerate(titles_urls[:num_results]):
            result = {
                "title": title.strip(),
                "url": url,
                "snippet": snippets[i].strip() if i < len(snippets) else ""
            }
            results.append(result)
        
        return results
        
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")
        return []


def _is_private_ip(ip_str: str) -> bool:
    """
    Check if an IP address is private, loopback, or link-local.
    
    Blocks:
    - Private ranges: 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
    - Loopback: 127.0.0.0/8, ::1
    - Link-local: 169.254.0.0/16, fe80::/10
    - Other reserved ranges
    """
    try:
        ip = ipaddress.ip_address(ip_str)
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        )
    except ValueError:
        # Invalid IP address - reject to be safe
        return True


def _validate_url_for_ssrf(url: str) -> tuple[bool, str]:
    """
    Validate a URL to prevent SSRF attacks.
    
    Returns:
        (is_safe, error_message) - is_safe is True if URL passes all checks
    """
    try:
        parsed = urlparse(url)
        
        # Only allow HTTPS
        if parsed.scheme != 'https':
            return False, f"Only HTTPS URLs are allowed, got: {parsed.scheme}"
        
        hostname = parsed.hostname
        if not hostname:
            return False, "URL has no hostname"
        
        # Resolve DNS to get IP addresses
        try:
            addr_info = socket.getaddrinfo(hostname, parsed.port or 443, proto=socket.IPPROTO_TCP)
        except socket.gaierror as e:
            return False, f"DNS resolution failed for {hostname}: {e}"
        
        if not addr_info:
            return False, f"No DNS records found for {hostname}"
        
        # Check all resolved IPs - block if any are private/internal
        for family, type_, proto, canonname, sockaddr in addr_info:
            ip_str = sockaddr[0]
            if _is_private_ip(ip_str):
                return False, f"URL resolves to private/internal IP: {ip_str}"
        
        return True, ""
        
    except Exception as e:
        return False, f"URL validation error: {e}"


def _fetch_page_content(url: str, max_length: int = 3000) -> str:
    """
    Fetch and extract text content from a URL.
    
    Includes SSRF protections:
    - Only HTTPS URLs allowed
    - DNS resolution checked against private/loopback/link-local IPs
    - Redirects disabled
    """
    if not REQUESTS_AVAILABLE:
        return ""
    
    # Validate URL for SSRF before fetching
    is_safe, error_msg = _validate_url_for_ssrf(url)
    if not is_safe:
        logger.warning(f"SSRF protection blocked URL {url}: {error_msg}")
        return ""
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; EdgeAI-Agent/1.0)'
        }
        # Disable redirects to prevent redirect-based SSRF bypasses
        response = requests.get(url, headers=headers, timeout=10, allow_redirects=False)
        
        # Handle redirects manually with validation
        if response.status_code in (301, 302, 303, 307, 308):
            redirect_url = response.headers.get('Location', '')
            if redirect_url:
                is_safe, error_msg = _validate_url_for_ssrf(redirect_url)
                if not is_safe:
                    logger.warning(f"SSRF protection blocked redirect to {redirect_url}: {error_msg}")
                    return ""
                # Fetch redirect target (single hop only)
                response = requests.get(redirect_url, headers=headers, timeout=10, allow_redirects=False)
        
        response.raise_for_status()
        
        return _extract_text_from_html(response.text, max_length)
        
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return ""


def web_search(
    query: str,
    num_results: int = 5,
    fetch_content: bool = False,
    focus: str = "general"
) -> Dict[str, Any]:
    """
    Search the web for information about ML models, frameworks, or related topics.
    
    Use this tool when you need more context about:
    - A specific ML model architecture or framework
    - Best practices for preprocessing or postprocessing
    - Model-specific documentation or papers
    - Troubleshooting inference issues
    
    Args:
        query: Search query string
        num_results: Maximum number of results to return (default: 5)
        fetch_content: Whether to fetch full page content for top results (slower)
        focus: Search focus - 'general', 'ml', 'documentation', 'github'
    
    Returns:
        Search results with titles, URLs, and snippets
    """
    try:
        if not query:
            return error_response(
                ValueError("query is required"),
                operation="web_search"
            )
        
        # Enhance query based on focus
        enhanced_query = query
        if focus == "ml":
            enhanced_query = f"machine learning {query}"
        elif focus == "documentation":
            enhanced_query = f"{query} documentation tutorial"
        elif focus == "github":
            enhanced_query = f"site:github.com {query}"
        
        # Perform search
        results = _search_duckduckgo(enhanced_query, num_results)
        
        if not results:
            # Return a helpful message if no results
            return ok(
                data={
                    "query": query,
                    "enhanced_query": enhanced_query,
                    "results": [],
                    "num_results": 0,
                    "note": "No search results found. Try rephrasing the query or check network connectivity."
                },
                message="No search results found"
            )
        
        # Optionally fetch content for top results
        if fetch_content and len(results) > 0:
            for i, result in enumerate(results[:2]):  # Only fetch first 2
                content = _fetch_page_content(result['url'])
                if content:
                    result['content_preview'] = content[:1500]
        
        # Build summary
        summary_parts = []
        for r in results[:3]:
            summary_parts.append(f"- {r['title']}: {r['snippet'][:100]}...")
        
        return ok(
            data={
                "query": query,
                "enhanced_query": enhanced_query if enhanced_query != query else None,
                "results": results,
                "num_results": len(results),
                "focus": focus
            },
            message=f"Found {len(results)} results for '{query}'"
        )
        
    except Exception as e:
        logger.error(f"Web search error: {e}", exc_info=True)
        return error_response(
            e,
            operation="web_search",
            query=query
        )


def search_model_info(model_name: str) -> Dict[str, Any]:
    """
    Search for information about a specific ML model.
    
    This is a specialized search that looks for:
    - Model architecture details
    - Input/output specifications
    - Preprocessing requirements
    - Common use cases
    
    Args:
        model_name: Name of the model to search for
    
    Returns:
        Aggregated information about the model from web sources
    """
    try:
        if not model_name:
            return error_response(
                ValueError("model_name is required"),
                operation="search_model_info"
            )
        
        # Clean model name for search
        clean_name = model_name.lower().replace('_', ' ').replace('-', ' ')
        
        # Identify model family
        model_families = {
            'yolo': 'YOLO object detection',
            'resnet': 'ResNet classification',
            'efficientnet': 'EfficientNet classification',
            'mobilenet': 'MobileNet classification',
            'deeplabv3': 'DeepLabV3 segmentation',
            'unet': 'U-Net segmentation',
            'bert': 'BERT language model',
            'vit': 'Vision Transformer',
            'ssd': 'SSD object detection',
            'faster rcnn': 'Faster R-CNN object detection',
            'mask rcnn': 'Mask R-CNN instance segmentation',
            'hrnet': 'HRNet pose estimation',
            'openpose': 'OpenPose pose estimation',
        }
        
        detected_family = None
        for family, description in model_families.items():
            if family in clean_name:
                detected_family = (family, description)
                break
        
        # Search for model-specific info
        search_query = f"{model_name} model input output preprocessing"
        results = _search_duckduckgo(search_query, 5)
        
        # Also search for GitHub/documentation
        doc_query = f"{model_name} github documentation"
        doc_results = _search_duckduckgo(doc_query, 3)
        
        # Combine and deduplicate
        all_results = results + [r for r in doc_results if r['url'] not in [x['url'] for x in results]]
        
        response_data = {
            "model_name": model_name,
            "detected_family": detected_family[1] if detected_family else None,
            "search_results": all_results[:8],
            "num_results": len(all_results)
        }
        
        # Add common knowledge based on model family
        if detected_family:
            family_key = detected_family[0]
            if 'yolo' in family_key:
                response_data["common_info"] = {
                    "type": "object_detection",
                    "typical_input": "RGB image, commonly 640x640 or 416x416",
                    "typical_output": "Bounding boxes with class and confidence",
                    "preprocessing": "Normalize to [0,1] or [-1,1], resize with letterboxing"
                }
            elif 'resnet' in family_key or 'efficientnet' in family_key or 'mobilenet' in family_key:
                response_data["common_info"] = {
                    "type": "classification",
                    "typical_input": "RGB image, commonly 224x224",
                    "typical_output": "Class probabilities (softmax)",
                    "preprocessing": "Normalize with ImageNet mean/std"
                }
            elif 'deeplab' in family_key or 'unet' in family_key:
                response_data["common_info"] = {
                    "type": "segmentation",
                    "typical_input": "RGB image",
                    "typical_output": "Per-pixel class masks",
                    "preprocessing": "Normalize, resize to model input size"
                }
        
        summary = f"Found information about {model_name}"
        if detected_family:
            summary += f" (detected as {detected_family[1]})"
        
        return ok(
            data=response_data,
            message=summary
        )
        
    except Exception as e:
        logger.error(f"Model info search error: {e}", exc_info=True)
        return error_response(
            e,
            operation="search_model_info",
            model_name=model_name
        )


# Register the tools
register_tool(
    name="web_search",
    func=web_search,
    description="Search the web for information about ML models, frameworks, preprocessing techniques, or any related topic. Use this when you need more context about a model or technique.",
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string"
            },
            "num_results": {
                "type": "integer",
                "default": 5,
                "description": "Maximum number of results to return"
            },
            "fetch_content": {
                "type": "boolean",
                "default": False,
                "description": "Whether to fetch full page content (slower but more detailed)"
            },
            "focus": {
                "type": "string",
                "enum": ["general", "ml", "documentation", "github"],
                "default": "general",
                "description": "Search focus area"
            }
        },
        "required": ["query"]
    }
)

register_tool(
    name="search_model_info",
    func=search_model_info,
    description="Search for detailed information about a specific ML model including architecture, preprocessing, and usage. Use this when you need to understand an unfamiliar model.",
    input_schema={
        "type": "object",
        "properties": {
            "model_name": {
                "type": "string",
                "description": "Name of the model to search for"
            }
        },
        "required": ["model_name"]
    }
)
