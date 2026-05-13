"""Agent chat routes for AI-powered model exploration."""

import logging
import os
import re
import time
import unicodedata
import traceback
import threading
import json
from typing import Any, Dict, List, Generator, Optional, Tuple

from flask import Blueprint, jsonify, request, Response, stream_with_context

from observability.request_context import (
    clear_request_context,
    new_request_id,
    set_request_context,
)
from observability.tracing import get_tracing

logger = logging.getLogger(__name__)


def _sanitize_filename(filename: str) -> str:
    """Normalize filenames to ASCII-safe, simple characters for storage and tool paths."""
    # Normalize unicode, drop non-ASCII, keep dots/underscores/dashes, collapse whitespace
    normalized = unicodedata.normalize("NFKD", filename)
    ascii_name = normalized.encode("ascii", "ignore").decode()
    ascii_name = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_name).strip("._-")
    return ascii_name[:128] or "upload"

# Create Blueprint
agent_bp = Blueprint('agent', __name__, url_prefix='/agent')

# Legacy in-memory session storage (kept for backward compatibility during transition)
# New code should use mcp.session.get_or_create_session() instead
agent_sessions: Dict[str, Dict[str, Any]] = {}
agent_sessions_lock = threading.Lock()

# Maximum concurrent sessions to prevent memory exhaustion
MAX_AGENT_SESSIONS = int(os.environ.get('MAX_AGENT_SESSIONS', '1000'))


def _infer_context_from_tool_calls(tool_calls: List[Dict[str, Any]]) -> str:
    """
    Infer the exploration context from tools that have been called.
    
    Args:
        tool_calls: List of tool call records from the session
        
    Returns:
        Context string for the recommend_next_steps tool
    """
    if not tool_calls:
        return "initial"
    
    tools_called = {tc.get('name', '') for tc in tool_calls}
    
    if 'get_frontend_integration_guide' in tools_called:
        return "ready_to_integrate"
    elif 'get_model_output_interpretation' in tools_called:
        return "checked_outputs"
    elif 'get_model_input_requirements' in tools_called:
        return "checked_inputs"
    elif 'analyze_model_type' in tools_called or 'get_model_metadata' in tools_called:
        return "analyzed_type"
    elif 'list_available_models' in tools_called:
        return "listed_models"
    
    return "initial"


_last_cleanup_time: float = 0.0
_CLEANUP_INTERVAL_SECONDS: float = 60.0


def _cleanup_old_sessions():
    """
    Clean up sessions with proper warning flow.

    Throttled to run at most once every 60 seconds to avoid expensive
    filesystem I/O on every chat request.

    Uses the new session tracking system which ensures:
    1. Sessions are warned before cleanup
    2. Grace period is given after warning
    3. Cleanup only occurs after grace period expires
    """
    global _last_cleanup_time
    now = time.time()
    if now - _last_cleanup_time < _CLEANUP_INTERVAL_SECONDS:
        return
    _last_cleanup_time = now

    try:
        from sessions.registry import (
            cleanup_inactive_sessions,
            get_session_registry,
            cleanup_session_storage,
        )
        
        # Use new tracking-based cleanup
        cleaned_up, pending_warnings = cleanup_inactive_sessions()
        
        for sid in cleaned_up:
            # Also remove from legacy storage if present
            with agent_sessions_lock:
                if sid in agent_sessions:
                    del agent_sessions[sid]
            logger.info(f"Cleaned up expired session (with warning flow): {sid}")
        
        # Log pending warnings (actual notification happens in response)
        if pending_warnings:
            logger.info(f"Sessions pending inactivity warning: {len(pending_warnings)}")
        
    except ImportError:
        # Fallback to legacy cleanup if new modules not available
        _cleanup_old_sessions_legacy()


def _cleanup_old_sessions_legacy():
    """Legacy cleanup for backward compatibility (no warning flow)."""
    current_time = time.time()
    cutoff_time = current_time - 3600  # 1 hour
    
    with agent_sessions_lock:
        sessions_to_remove = [
            sid for sid, session_data in agent_sessions.items()
            if session_data.get('last_activity', 0) < cutoff_time
        ]
        for sid in sessions_to_remove:
            del agent_sessions[sid]
            logger.info(f"Cleaned up expired session (legacy): {sid}")
            
            # Also cleanup session storage
            try:
                from agents.tools import cleanup_session_storage
                cleanup_session_storage(sid)
            except Exception as e:
                logger.warning(f"Failed to cleanup storage for session {sid}: {e}")


def _get_session_warnings(session_id: str) -> Dict[str, Any]:
    """
    Get any warnings that should be included in the response.
    
    Args:
        session_id: Session identifier
    
    Returns:
        Dictionary with warning information to include in response
    """
    try:
        from sessions.registry import check_session_warnings, is_session_over_hard_limit
        from sessions.tracking import WarningLevel
        
        usage_warnings, inactivity_warning = check_session_warnings(session_id)
        
        warnings_data = {
            "has_warnings": False,
            "usage_warnings": [],
            "inactivity_warning": None,
            "hard_limit_exceeded": False,
            "exceeded_dimension": None,
        }
        
        if usage_warnings:
            warnings_data["has_warnings"] = True
            warnings_data["usage_warnings"] = [w.to_dict() for w in usage_warnings]
            
            # Check if any are hard limit exceeded
            for w in usage_warnings:
                if w.level == WarningLevel.EXCEEDED:
                    warnings_data["hard_limit_exceeded"] = True
                    warnings_data["exceeded_dimension"] = w.dimension.value
                    break
        
        if inactivity_warning:
            warnings_data["has_warnings"] = True
            warnings_data["inactivity_warning"] = inactivity_warning.to_dict()
        
        # Double-check hard limits
        exceeded, dimension = is_session_over_hard_limit(session_id)
        if exceeded:
            warnings_data["hard_limit_exceeded"] = True
            warnings_data["exceeded_dimension"] = dimension.value if dimension else None
        
        return warnings_data
        
    except ImportError:
        return {"has_warnings": False, "usage_warnings": [], "inactivity_warning": None}


def _format_warning_message(warnings_data: Dict[str, Any]) -> Optional[str]:
    """
    Format warnings into a user-facing message to prepend to responses.
    
    Args:
        warnings_data: Warning data from _get_session_warnings
    
    Returns:
        Formatted warning message or None
    """
    if not warnings_data.get("has_warnings"):
        return None
    
    messages = []
    
    # Add usage warnings
    for w in warnings_data.get("usage_warnings", []):
        messages.append(w.get("message", ""))
    
    # Add inactivity warning
    inactivity = warnings_data.get("inactivity_warning")
    if inactivity:
        messages.append(inactivity.get("message", ""))
    
    if messages:
        return "\n\n".join(filter(None, messages)) + "\n\n---\n\n"
    
    return None


def _record_session_usage(
    session_id: str,
    tokens: Optional[Dict[str, int]] = None,
    image_created: bool = False,
) -> None:
    """
    Record usage metrics for a session.
    
    Args:
        session_id: Session identifier
        tokens: Token usage dict with 'prompt_tokens' and 'completion_tokens'
        image_created: Whether an image was created in this request
    """
    try:
        from sessions.registry import get_session
        
        session = get_session(session_id)
        if session is None:
            return
        
        # Record request
        session.record_request()
        
        # Record tokens if provided
        if tokens:
            session.record_tokens(
                prompt=tokens.get('prompt_tokens', 0),
                completion=tokens.get('completion_tokens', 0),
            )
        
        # Record image if created
        if image_created:
            session.record_image()
            
    except ImportError:
        pass  # New tracking not available


@agent_bp.route('/chat', methods=['POST'])
def agent_chat():
    """
    AI agent chat endpoint for conversational model exploration.
    
    Request body (JSON or multipart/form-data):
        JSON:
        {
            "message": "User message",
            "session_id": "Optional session ID for conversation continuity"
        }
        
        Multipart (with image):
        - message: User message
        - session_id: Optional session ID
        - image: Image file
    
    Response:
        {
            "success": true,
            "response": "Agent response",
            "session_id": "Session ID for next request",
            "enabled": true/false,
            "tool_calls": [...],
            "tokens": {...}
        }
    """
    try:
        # Import agent modules (lazy import to avoid issues if anthropic not installed)
        try:
            from agents.prompts import process_chat_message, check_agent_enabled
            from agents.tools import get_session_storage_path, check_session_storage_limit
        except ImportError as e:
            logger.error(f"Failed to import agent modules: {e}")
            return jsonify({
                "success": False,
                "enabled": False,
                "error": "Agent modules not available. Check server logs.",
                "response": "⚠️ Agent is not available. Please contact administrator."
            }), 500
        
        # Check if agent is enabled
        if not check_agent_enabled():
            return jsonify({
                "success": False,
                "enabled": False,
                "response": "⚠️ AI Agent is not configured. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, or LLM_SERVER_URL.",
                "message": "Configure an API key to enable agent"
            }), 200
        
        # Parse request (either JSON or multipart form data)
        image_path = None
        if request.content_type and 'multipart/form-data' in request.content_type:
            user_message = request.form.get('message')
            session_id = request.form.get('session_id') or f"session_{int(time.time() * 1000)}"
            
            if not user_message:
                return jsonify({
                    "success": False,
                    "error": "Missing 'message' in form data"
                }), 400
            
            # Debug: Log session info
            logger.info(f"📝 Agent chat request (multipart) - session_id: {session_id}, message preview: {user_message[:50]}...")
            
            # Handle image if present
            if 'image' in request.files:
                image_file = request.files['image']
                if image_file.filename:
                    within_limit, current_mb = check_session_storage_limit(session_id)
                    if not within_limit:
                        return jsonify({
                            "success": False,
                            "error": f"Session storage limit exceeded ({current_mb:.1f}MB / 30MB).",
                            "response": f"⚠️ Your session has exceeded the 30MB storage limit."
                        }), 413
                    
                    session_dir = get_session_storage_path(session_id)
                    timestamp = int(time.time() * 1000)
                    safe_name = _sanitize_filename(image_file.filename)
                    image_filename = f"upload_{timestamp}_{safe_name}"
                    image_path = os.path.join(session_dir, image_filename)
                    image_file.save(image_path)
                    logger.info(f"Saved uploaded image to {image_path} (original name: {image_file.filename})")
        else:
            data = request.get_json()
            if not data or 'message' not in data:
                return jsonify({
                    "success": False,
                    "error": "Missing 'message' in request body"
                }), 400
            
            user_message = data['message']
            session_id = data.get('session_id') or f"session_{int(time.time() * 1000)}"

            logger.info(f"📝 Agent chat request (json) - session_id: {session_id}, message preview: {user_message[:50]}...")

        # Clean up old sessions periodically
        _cleanup_old_sessions()
        
        # Initialize session in new tracking system
        image_created = image_path is not None
        try:
            from sessions.registry import get_or_create_session, SessionCapacityError
            tracked_session, is_new = get_or_create_session(session_id)
            
            # Touch session to record activity (resets inactivity warning if user responds)
            tracked_session.touch()
            
            # Check for hard limit exceeded before processing
            warnings_data = _get_session_warnings(session_id)
            if warnings_data.get("hard_limit_exceeded"):
                dimension = warnings_data.get("exceeded_dimension", "usage")
                return jsonify({
                    "success": False,
                    "error": f"Session {dimension} limit exceeded",
                    "response": f"⚠️ Your session has exceeded the {dimension} limit. Please start a new session to continue.",
                    "session_id": session_id,
                    "warnings": warnings_data,
                }), 429
                
        except SessionCapacityError as e:
            return jsonify({
                "success": False,
                "error": "Too many active sessions. Please try again later.",
                "response": "⚠️ Server is at capacity. Please try again in a few minutes."
            }), 503
        except ImportError:
            # Fall back to legacy tracking
            warnings_data = {"has_warnings": False}
        
        # Get or create session (legacy tracking for backward compatibility)
        with agent_sessions_lock:
            if session_id not in agent_sessions:
                # Prevent memory exhaustion by limiting concurrent sessions
                if len(agent_sessions) >= MAX_AGENT_SESSIONS:
                    return jsonify({
                        "success": False,
                        "error": "Too many active sessions. Please try again later.",
                        "response": "⚠️ Server is at capacity. Please try again in a few minutes."
                    }), 503
                agent_sessions[session_id] = {
                    'history': [],
                    'tool_calls': [],
                    'current_model': None,
                    'exploration_context': 'initial',
                    'created_at': time.time(),
                    'last_activity': time.time()
                }
            
            session = agent_sessions[session_id]
            # Copy history so the LLM gets a stable snapshot; avoids race
            # conditions if another concurrent request mutates the list.
            history = list(session['history'])
            session['last_activity'] = time.time()

        # Process message with LLM, instrumented with Langfuse trace if enabled.
        request_id = new_request_id()
        tokens = set_request_context(request_id=request_id, session_id=session_id)
        tracing = get_tracing()
        try:
            with tracing.chat_turn(
                session_id=session_id,
                request_id=request_id,
                user_metadata={
                    "endpoint": "/agent/chat",
                    "has_image": bool(image_path),
                    "history_len": len(history),
                },
            ):
                result = process_chat_message(
                    user_message,
                    history,
                    session_id=session_id,
                    image_path=image_path
                )
        finally:
            if tracing.enabled:
                tracing.flush()
            clear_request_context(tokens)
        
        if result is None:
            result = {
                "success": False,
                "error": "No response from agent",
                "response": "Sorry, I encountered an internal error. Please try again.",
                "enabled": True
            }
        
        # Update session history if successful
        if result.get('success'):
            with agent_sessions_lock:
                history.append({
                    "role": "user",
                    "content": user_message
                })
                
                # Build assistant content that includes tool call summaries
                assistant_content = result.get('response', '') or ''
                logger.info(f"📝 Building history - response length: {len(assistant_content)}, preview: {assistant_content[:100] if assistant_content else 'EMPTY'}...")
                
                # Include tool call context in the stored history so the model remembers what it did
                tool_calls = result.get('tool_calls', [])
                if tool_calls:
                    tool_summary_parts = []
                    for tc in tool_calls:
                        tool_name = tc.get('name', 'unknown') if isinstance(tc, dict) else 'unknown'
                        tool_args = tc.get('arguments', {}) if isinstance(tc, dict) else {}
                        tool_result = tc.get('result', {}) if isinstance(tc, dict) else {}
                        
                        # Ensure tool_args and tool_result are dicts
                        if isinstance(tool_args, str):
                            try:
                                tool_args = json.loads(tool_args)
                            except (json.JSONDecodeError, ValueError):
                                tool_args = {}
                        if isinstance(tool_result, str):
                            try:
                                tool_result = json.loads(tool_result)
                            except (json.JSONDecodeError, ValueError):
                                tool_result = {}
                        
                        # Create a concise summary
                        if tool_name == 'list_available_models':
                            models = tool_result.get('models', []) if isinstance(tool_result, dict) else []
                            model_names = [m.get('name', 'unknown') if isinstance(m, dict) else str(m) for m in models] if models else []
                            tool_summary_parts.append(f"[Called {tool_name}: Found models: {', '.join(model_names) if model_names else 'none'}]")
                        elif tool_name == 'run_inference':
                            model_used = tool_args.get('model_name', 'unknown') if isinstance(tool_args, dict) else 'unknown'
                            tool_summary_parts.append(f"[Called {tool_name} with model={model_used}]")
                        elif tool_name == 'get_model_metadata':
                            model_used = tool_args.get('model_name', 'unknown') if isinstance(tool_args, dict) else 'unknown'
                            tool_summary_parts.append(f"[Called {tool_name} for {model_used}]")
                        elif tool_name == 'analyze_model_type':
                            model_used = tool_args.get('model_name', 'unknown') if isinstance(tool_args, dict) else 'unknown'
                            model_type = tool_result.get('detected_type', 'unknown') if isinstance(tool_result, dict) else 'unknown'
                            tool_summary_parts.append(f"[Called {tool_name}: {model_used} is type={model_type}]")
                        else:
                            tool_summary_parts.append(f"[Called {tool_name}]")
                    
                    if tool_summary_parts:
                        assistant_content = "\n".join(tool_summary_parts) + "\n\n" + assistant_content
                
                history.append({
                    "role": "assistant",
                    "content": assistant_content
                })
                
                if tool_calls:
                    session['tool_calls'].extend(tool_calls)
                    session['exploration_context'] = _infer_context_from_tool_calls(session['tool_calls'])
                    
                    for tc in tool_calls:
                        model_name = tc.get('arguments', {}).get('model_name') or tc.get('input', {}).get('model_name')
                        if model_name:
                            session['current_model'] = model_name
                
                # Keep only last 20 messages
                if len(history) > 20:
                    agent_sessions[session_id]['history'] = history[-20:]
                
                # Keep only last 50 tool calls
                if len(session['tool_calls']) > 50:
                    session['tool_calls'] = session['tool_calls'][-50:]
        
        # Record usage in new tracking system
        tokens_data = result.get('tokens', {})
        _record_session_usage(
            session_id,
            tokens=tokens_data if tokens_data else None,
            image_created=image_created,
        )
        
        # Get updated warnings after recording usage
        warnings_data = _get_session_warnings(session_id)
        
        # Build response
        response_text = result.get('response', '')
        
        # Prepend warning message if there are warnings
        warning_message = _format_warning_message(warnings_data)
        if warning_message and response_text:
            response_text = warning_message + response_text
        
        response_data = {
            "success": result.get('success', False),
            "response": response_text,
            "session_id": session_id,
            "enabled": result.get('enabled', True)
        }
        
        if 'tool_calls' in result:
            response_data['tool_calls'] = result['tool_calls']
        if 'tokens' in result:
            response_data['tokens'] = result['tokens']
        if 'error' in result:
            response_data['error'] = result['error']
        
        # Add warnings to response if present
        if warnings_data.get("has_warnings"):
            response_data['warnings'] = warnings_data
        
        # Add context info (including new metrics)
        with agent_sessions_lock:
            if session_id in agent_sessions:
                context_data = {
                    'exploration_state': agent_sessions[session_id].get('exploration_context', 'initial'),
                    'current_model': agent_sessions[session_id].get('current_model'),
                    'tools_used_count': len(agent_sessions[session_id].get('tool_calls', []))
                }
                
                # Add metrics from new tracking system
                try:
                    from sessions.registry import get_session_status
                    session_status = get_session_status(session_id)
                    if session_status:
                        context_data['metrics'] = session_status.get('metrics', {})
                        context_data['warning_state'] = session_status.get('warning_state', {})
                except ImportError:
                    pass
                
                response_data['context'] = context_data
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error in agent chat endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "enabled": True,
            "error": "Internal server error",
            "response": "Sorry, I encountered an internal error. Please try again."
        }), 500


def _generate_sse_events(message: str, session_id: str, image_path: str = None) -> Generator[str, None, None]:
    """
    Generate SSE events for streaming chat responses.
    
    Uses true streaming when available via the LLM router.
    Falls back to non-streaming atomic response when streaming is not supported.
    
    Yields SSE-formatted events with types:
    - start: Initial connection established, includes session_id
    - warning: Usage or inactivity warnings
    - token: Individual text token (or chunk)
    - tool_start: Tool execution beginning
    - tool_end: Tool execution completed with result
    - done: Final message with complete response and metadata
    - error: Error occurred
    """
    try:
        from agents.prompts import process_chat_message_stream, check_agent_enabled
        from agents.tools import get_session_storage_path
    except ImportError as e:
        logger.error(f"Failed to import agent modules: {e}")
        yield f"event: error\ndata: {json.dumps({'error': 'Agent modules not available'})}\n\n"
        return
    
    if not check_agent_enabled():
        yield f"event: error\ndata: {json.dumps({'error': 'Agent not configured', 'enabled': False})}\n\n"
        return
    
    # Initialize session in new tracking system
    image_created = image_path is not None
    try:
        from sessions.registry import get_or_create_session, SessionCapacityError
        tracked_session, is_new = get_or_create_session(session_id)
        
        # Touch session to record activity (resets inactivity warning if user responds)
        tracked_session.touch()
        
        # Check for hard limit exceeded before processing
        warnings_data = _get_session_warnings(session_id)
        if warnings_data.get("hard_limit_exceeded"):
            dimension = warnings_data.get("exceeded_dimension", "usage")
            yield f"event: error\ndata: {json.dumps({'error': f'Session {dimension} limit exceeded', 'limit_exceeded': True})}\n\n"
            return
            
    except SessionCapacityError as e:
        yield f"event: error\ndata: {json.dumps({'error': 'Server at capacity'})}\n\n"
        return
    except ImportError:
        warnings_data = {"has_warnings": False}
    
    # Get or create session (legacy tracking)
    with agent_sessions_lock:
        if session_id not in agent_sessions:
            agent_sessions[session_id] = {
                'history': [],
                'tool_calls': [],
                'current_model': None,
                'exploration_context': 'initial',
                'created_at': time.time(),
                'last_activity': time.time()
            }
        session = agent_sessions[session_id]
        history = list(session['history'])  # Copy to avoid mutation during iteration
        session['last_activity'] = time.time()
    
    # Send start event with any initial warnings
    start_data = {'session_id': session_id}
    if warnings_data.get("has_warnings"):
        start_data['warnings'] = warnings_data
    yield f"event: start\ndata: {json.dumps(start_data, ensure_ascii=False)}\n\n"
    
    # Send warning event if there are warnings (separate event for visibility)
    if warnings_data.get("has_warnings"):
        yield f"event: warning\ndata: {json.dumps(warnings_data, ensure_ascii=False)}\n\n"
    
    # Wrap the full streaming turn in a Langfuse trace (no-op when disabled).
    request_id = new_request_id()
    ctx_tokens = set_request_context(request_id=request_id, session_id=session_id)
    tracing = get_tracing()
    chat_span_cm = tracing.chat_turn(
        session_id=session_id,
        request_id=request_id,
        user_metadata={
            "endpoint": "/agent/chat/stream",
            "has_image": bool(image_path),
            "history_len": len(history),
        },
    )

    try:
        chat_span_cm.__enter__()
        # Use true streaming via process_chat_message_stream
        full_response = ""
        tool_calls_made = []
        tokens_data = {}

        for event in process_chat_message_stream(message, history, session_id=session_id, image_path=image_path):
            event_type = event.get("type")
            
            if event_type == "token":
                token = event.get("content", "")
                full_response += token
                yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
            
            elif event_type == "tool_start":
                yield f"event: tool_start\ndata: {json.dumps({'name': event.get('name', ''), 'id': event.get('id', '')}, ensure_ascii=False)}\n\n"
            
            elif event_type == "tool_end":
                tc_data = {
                    "name": event.get("name", ""),
                    "result": event.get("result", {})
                }
                tool_calls_made.append(tc_data)
                yield f"event: tool_end\ndata: {json.dumps(tc_data, ensure_ascii=False)}\n\n"
            
            elif event_type == "done":
                full_response = event.get("response", full_response)
                tool_calls_made = event.get("tool_calls", tool_calls_made)
                tokens_data = event.get("meta", {}).get("tokens", {})
                
                # Record usage in new tracking system
                _record_session_usage(
                    session_id,
                    tokens=tokens_data if tokens_data else None,
                    image_created=image_created,
                )
                
                # Update session history
                with agent_sessions_lock:
                    if session_id in agent_sessions:
                        agent_sessions[session_id]['history'].append({"role": "user", "content": message})
                        agent_sessions[session_id]['history'].append({"role": "assistant", "content": full_response})
                        
                        if tool_calls_made:
                            agent_sessions[session_id]['tool_calls'].extend(tool_calls_made)
                            agent_sessions[session_id]['exploration_context'] = _infer_context_from_tool_calls(
                                agent_sessions[session_id]['tool_calls']
                            )
                        
                        # Limit history size
                        if len(agent_sessions[session_id]['history']) > 20:
                            agent_sessions[session_id]['history'] = agent_sessions[session_id]['history'][-20:]
                
                # Get updated warnings after processing
                updated_warnings = _get_session_warnings(session_id)
                
                # Send done event
                done_data = {
                    'response': full_response,
                    'session_id': session_id,
                    'tool_calls': tool_calls_made,
                    'success': True,
                    'finish_reason': event.get("finish_reason"),
                    'meta': event.get("meta", {})
                }
                if updated_warnings.get("has_warnings"):
                    done_data['warnings'] = updated_warnings
                yield f"event: done\ndata: {json.dumps(done_data, ensure_ascii=False)}\n\n"
            
            elif event_type == "complete":
                # Non-streaming atomic response - render complete response at once
                # No cursor, no typing animation - just the complete response
                full_response = event.get("response", "")
                tool_calls_made = event.get("tool_calls", [])
                tokens_data = event.get("meta", {}).get("tokens", {})
                
                # Record usage in new tracking system
                _record_session_usage(
                    session_id,
                    tokens=tokens_data if tokens_data else None,
                    image_created=image_created,
                )
                
                # Update session history
                with agent_sessions_lock:
                    if session_id in agent_sessions:
                        agent_sessions[session_id]['history'].append({"role": "user", "content": message})
                        agent_sessions[session_id]['history'].append({"role": "assistant", "content": full_response})
                        
                        if tool_calls_made:
                            agent_sessions[session_id]['tool_calls'].extend(tool_calls_made)
                            agent_sessions[session_id]['exploration_context'] = _infer_context_from_tool_calls(
                                agent_sessions[session_id]['tool_calls']
                            )
                        
                        # Limit history size
                        if len(agent_sessions[session_id]['history']) > 20:
                            agent_sessions[session_id]['history'] = agent_sessions[session_id]['history'][-20:]
                
                # Get updated warnings
                updated_warnings = _get_session_warnings(session_id)
                
                # Send complete event (non-streaming mode)
                complete_data = {
                    'response': full_response,
                    'session_id': session_id,
                    'tool_calls': tool_calls_made,
                    'success': True,
                    'streaming': False,
                    'finish_reason': event.get("finish_reason"),
                    'meta': event.get("meta", {})
                }
                if updated_warnings.get("has_warnings"):
                    complete_data['warnings'] = updated_warnings
                yield f"event: complete\ndata: {json.dumps(complete_data, ensure_ascii=False)}\n\n"
            
            elif event_type == "error":
                err_payload: Dict[str, Any] = {
                    "error": event.get("error", "Unknown error"),
                }
                # Forward rate-limit metadata from the upstream adapter so
                # the frontend can render a retry-window message.
                for key in ("retry_after", "status_code", "error_code"):
                    if event.get(key) is not None:
                        err_payload[key] = event[key]
                yield f"event: error\ndata: {json.dumps(err_payload, ensure_ascii=False)}\n\n"
                return
        
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        logger.error(traceback.format_exc())
        yield f"event: error\ndata: {json.dumps({'error': 'Internal server error'}, ensure_ascii=False)}\n\n"
    finally:
        try:
            chat_span_cm.__exit__(None, None, None)
        except Exception:
            pass
        if tracing.enabled:
            try:
                tracing.flush()
            except Exception:
                pass
        clear_request_context(ctx_tokens)


@agent_bp.route('/chat/stream', methods=['POST'])
def agent_chat_stream():
    """
    Streaming AI agent chat endpoint using Server-Sent Events (SSE).

    Accepts both JSON and multipart/form-data so a single client-side
    path handles text-only chats and image-upload chats alike. The
    legacy non-streaming /agent/chat endpoint remains for backward
    compatibility but the frontend no longer needs it.

    Request bodies:
        JSON:       { message, session_id? }
        Multipart:  message, session_id?, image (file)

    File uploads are saved under the session storage dir; only the
    server-side resolved path is used downstream — we never honor an
    arbitrary file path from the JSON body.

    Response: text/event-stream with events:
        - event: start - Connection established
        - event: token - Individual text token
        - event: tool_start - Tool execution starting
        - event: tool_end - Tool execution completed
        - event: done - Final response complete
        - event: error - Error occurred
    """
    try:
        image_path = None
        if request.content_type and 'multipart/form-data' in request.content_type:
            message = request.form.get('message')
            session_id = request.form.get('session_id') or f"session_{int(time.time() * 1000)}"

            if not message:
                return jsonify({"error": "Missing 'message' in form data"}), 400

            if 'image' in request.files:
                image_file = request.files['image']
                if image_file.filename:
                    try:
                        from sessions.registry import (
                            check_session_storage_limit,
                            get_session_storage_path,
                        )
                    except ImportError:
                        check_session_storage_limit = None  # type: ignore[assignment]
                        get_session_storage_path = None  # type: ignore[assignment]

                    if check_session_storage_limit is not None:
                        within_limit, current_mb = check_session_storage_limit(session_id)
                        if not within_limit:
                            return jsonify({
                                "error": (
                                    f"Session storage limit exceeded ({current_mb:.1f}MB / 30MB)."
                                ),
                            }), 413

                    if get_session_storage_path is not None:
                        session_dir = get_session_storage_path(session_id)
                        timestamp = int(time.time() * 1000)
                        safe_name = _sanitize_filename(image_file.filename)
                        image_filename = f"upload_{timestamp}_{safe_name}"
                        image_path = os.path.join(session_dir, image_filename)
                        image_file.save(image_path)
                        logger.info(
                            "Saved uploaded image (stream path) to %s (original name: %s)",
                            image_path, image_file.filename,
                        )
        else:
            data = request.get_json(silent=True) or {}
            if 'message' not in data:
                return jsonify({"error": "Missing 'message' in request body"}), 400

            message = data['message']
            session_id = data.get('session_id') or f"session_{int(time.time() * 1000)}"

        _cleanup_old_sessions()
        
        # Use content_type= (raw header passthrough). Passing mimetype=
        # with an explicit charset makes Flask append its own default
        # charset on top, producing `charset=utf-8; charset=utf-8` —
        # some browsers reject that as malformed SSE.
        return Response(
            stream_with_context(_generate_sse_events(message, session_id, image_path)),
            content_type='text/event-stream; charset=utf-8',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
            }
        )
        
    except Exception as e:
        logger.error(f"Error in streaming endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500


@agent_bp.route('/status')
def agent_status():
    """Check if agent is enabled and get status info."""
    try:
        from agents.prompts import check_agent_enabled, get_backend_info
        
        backend_info = get_backend_info()
        enabled = backend_info.get("enabled", False)
        
        with agent_sessions_lock:
            active_sessions = len(agent_sessions)
        
        available_tools = [
            {"name": "list_available_models", "description": "Discover deployed models"},
            {"name": "get_model_metadata", "description": "Get model specifications"},
            {"name": "analyze_model_type", "description": "Infer model type from shapes"},
            {"name": "get_model_input_requirements", "description": "Get preprocessing guidance"},
            {"name": "get_model_output_interpretation", "description": "Get post-processing guidance"},
            {"name": "get_server_status", "description": "Check server health"},
            {"name": "get_api_examples", "description": "Get API examples and curl commands"},
            {"name": "get_frontend_integration_guide", "description": "Get integration code examples"},
            {"name": "recommend_next_steps", "description": "Get suggested next actions"},
        ]
        
        # Check if LLM router is enabled
        llm_router_info = {}
        try:
            from router import get_router
            router = get_router()
            active_provider = router.get_active_provider()
            llm_router_info = {
                "enabled": True,
                "providers": len(router.list_providers()),
                "active_provider": active_provider.get("name") if active_provider else None,
                "routing_strategy": router._routing_strategy.value
            }
        except Exception:
            llm_router_info = {"enabled": False, "providers": 0}
        
        return jsonify({
            "enabled": enabled,
            "active_sessions": active_sessions,
            "backend": backend_info.get("backend"),
            "model": backend_info.get("model"),
            "message": backend_info.get("message", "Agent is not configured"),
            "available_tools": available_tools if enabled else [],
            "supported_model_types": ["classification", "object_detection", "segmentation", "pose", "embedding"],
            "llm_router": llm_router_info
        })
    except Exception as e:
        return jsonify({
            "enabled": False,
            "error": str(e)
        }), 500


@agent_bp.route('/session/<session_id>/status', methods=['GET'])
def get_session_status(session_id: str):
    """
    Get detailed status for a specific session.
    
    Returns usage metrics, warning state, and session information.
    
    Response:
        {
            "success": true,
            "session_id": "...",
            "metrics": {
                "total_tokens": 1234,
                "image_count": 5,
                "request_count": 10,
                ...
            },
            "warnings": {...},
            "exists": true
        }
    """
    try:
        from sessions.registry import get_session_status as mcp_get_session_status
        
        status = mcp_get_session_status(session_id)
        
        if status is None:
            return jsonify({
                "success": True,
                "session_id": session_id,
                "exists": False,
                "message": "Session not found or expired"
            }), 200
        
        # Get current warnings
        warnings_data = _get_session_warnings(session_id)
        
        return jsonify({
            "success": True,
            "session_id": session_id,
            "exists": True,
            "metrics": status.get("metrics", {}),
            "warning_state": status.get("warning_state", {}),
            "warnings": warnings_data,
            "storage_mb": status.get("storage_mb", 0),
            "storage_image_count": status.get("storage_image_count", 0),
        })
        
    except ImportError:
        # Fall back to legacy session info
        with agent_sessions_lock:
            session = agent_sessions.get(session_id)
            if session is None:
                return jsonify({
                    "success": True,
                    "session_id": session_id,
                    "exists": False,
                    "message": "Session not found or expired"
                }), 200
            
            return jsonify({
                "success": True,
                "session_id": session_id,
                "exists": True,
                "metrics": {
                    "tool_call_count": len(session.get("tool_calls", [])),
                    "created_at": session.get("created_at"),
                    "last_activity": session.get("last_activity"),
                },
                "warning_state": {},
                "warnings": {"has_warnings": False},
            })
    
    except Exception as e:
        logger.error(f"Error getting session status: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@agent_bp.route('/session/<session_id>/keepalive', methods=['POST'])
def session_keepalive(session_id: str):
    """
    Keep a session alive by updating activity timestamp.
    
    This endpoint should be called by clients when they receive an
    inactivity warning and want to keep the session active.
    
    Response:
        {
            "success": true,
            "session_id": "...",
            "message": "Session kept alive"
        }
    """
    try:
        from sessions.registry import get_session
        
        session = get_session(session_id)
        if session is None:
            return jsonify({
                "success": False,
                "session_id": session_id,
                "error": "Session not found or expired"
            }), 404
        
        # Touch session to reset inactivity warnings
        session.touch()
        
        # Also update legacy session
        with agent_sessions_lock:
            if session_id in agent_sessions:
                agent_sessions[session_id]['last_activity'] = time.time()
        
        return jsonify({
            "success": True,
            "session_id": session_id,
            "message": "Session kept alive",
            "last_activity": session.metrics.last_activity,
        })
        
    except ImportError:
        # Fall back to legacy session update
        with agent_sessions_lock:
            if session_id not in agent_sessions:
                return jsonify({
                    "success": False,
                    "session_id": session_id,
                    "error": "Session not found or expired"
                }), 404
            
            agent_sessions[session_id]['last_activity'] = time.time()
            
            return jsonify({
                "success": True,
                "session_id": session_id,
                "message": "Session kept alive",
                "last_activity": agent_sessions[session_id]['last_activity'],
            })
    
    except Exception as e:
        logger.error(f"Error in session keepalive: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@agent_bp.route('/session/<session_id>/acknowledge-warnings', methods=['POST'])
def acknowledge_session_warnings(session_id: str):
    """
    Acknowledge current session warnings.
    
    This allows the user to continue despite soft warnings.
    
    Response:
        {
            "success": true,
            "session_id": "...",
            "message": "Warnings acknowledged"
        }
    """
    try:
        from sessions.registry import get_session
        
        session = get_session(session_id)
        if session is None:
            return jsonify({
                "success": False,
                "session_id": session_id,
                "error": "Session not found or expired"
            }), 404
        
        # Acknowledge warnings
        with session._lock:
            session.warning_state.acknowledge_warnings()
        
        # Also touch session
        session.touch()
        
        return jsonify({
            "success": True,
            "session_id": session_id,
            "message": "Warnings acknowledged",
        })
        
    except ImportError:
        return jsonify({
            "success": True,
            "session_id": session_id,
            "message": "Warnings acknowledged (tracking not available)",
        })
    
    except Exception as e:
        logger.error(f"Error acknowledging warnings: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@agent_bp.route('/session/config', methods=['GET'])
def get_session_configuration():
    """
    Get the current session configuration.
    
    Returns the configured limits and thresholds for sessions.
    
    Response:
        {
            "success": true,
            "config": {
                "limits": {...},
                "warnings": {...},
                "inactivity": {...}
            }
        }
    """
    try:
        from sessions.config import get_session_config
        
        config = get_session_config()
        
        return jsonify({
            "success": True,
            "config": config.to_dict(),
        })
        
    except ImportError:
        # Return minimal legacy config
        return jsonify({
            "success": True,
            "config": {
                "limits": {
                    "max_storage_mb": 30.0,
                },
                "session": {
                    "max_concurrent_sessions": MAX_AGENT_SESSIONS,
                }
            },
        })
    
    except Exception as e:
        logger.error(f"Error getting session config: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
