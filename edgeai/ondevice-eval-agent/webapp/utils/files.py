"""File handling utilities."""

from typing import Optional, Set


def allowed_file(filename: str, allowed_extensions: Optional[Set[str]] = None) -> bool:
    """Check if a file has an allowed extension.
    
    Args:
        filename: The filename to check
        allowed_extensions: Set of allowed extensions. If None, uses default set.
    
    Returns:
        True if the file extension is allowed, False otherwise.
    """
    if allowed_extensions is None:
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def get_class_name(class_id: int, model_name: Optional[str] = None) -> str:
    """Get class name for a given class ID.
    
    Class names are managed by the user via the UI (JSON file upload).
    This function returns a generic class identifier, with special handling
    for known single-class model types.
    The frontend will apply custom class names from user-uploaded JSON.
    
    Args:
        class_id: The numeric class ID from the model
        model_name: Optional model name for special handling
    
    Returns:
        Class name string (e.g., "face" for face detection models, "Class_0" otherwise)
    """
    # Check for face detection models
    if model_name:
        model_lower = model_name.lower()
        if any(kw in model_lower for kw in ['face', 'widerface', 'wider_face']):
            if class_id == 0:
                return "face"
    
    # Return generic class identifier - frontend will apply custom names
    return f"Class_{class_id}"
