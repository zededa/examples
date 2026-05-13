"""
Run Inference Tool

Allows the agent to run inference on images using deployed models.
Automatically detects model type and selects appropriate processing.
Uses a dedicated LLM instance to generate rich explanations of results.
"""

import logging
import os
import json
import base64
from typing import Dict, Any, Optional

from tools.base import ok, error_response, get_client
from tools.registry import register_tool
from sessions.registry import SESSION_STORAGE_ROOT

logger = logging.getLogger(__name__)

# Available processing types
AVAILABLE_PROCESSORS = [
    'classification',
    'detection', 
    'segmentation',
    'pose',
    'keypoint',
    'panoptic',
    'ocr',
    'auto'  # Auto-detect based on model
]


def _generate_llm_explanation(
    inference_data: Dict[str, Any],
    image_base64: Optional[str] = None,
    result_image_base64: Optional[str] = None
) -> str:
    """
    Use a dedicated LLM instance to generate a rich explanation of inference results.
    
    This creates a separate LLM call with fresh context, avoiding context overflow
    in the main agent conversation. If vision is supported and images are provided,
    the LLM can actually "see" what was in the image and the results.
    
    Args:
        inference_data: The structured inference results (without base64 images)
        image_base64: Optional base64 of the original image
        result_image_base64: Optional base64 of the result visualization
        
    Returns:
        A detailed explanation string from the LLM
    """
    try:
        # Try to get the router for LLM access
        from router import get_router
        import time
        router = get_router()
        
        active = router.get_active_provider()
        if not active or not active.get('status', {}).get('available', False):
            logger.warning("No active LLM provider for explanation generation")
            return None
        
        # Check if provider supports vision - check multiple possible fields
        # Also check model name for common vision model patterns
        supports_vision = (
            active.get('supports_vision', False) or 
            active.get('capabilities', {}).get('vision', False)
        )
        
        # Auto-detect vision capability from model name
        model_name_lower = active.get('model', '').lower()
        vision_keywords = ['vision', 'vl', 'visual', 'llava', 'gpt-4o', 'claude-3', 'gemini']
        if any(kw in model_name_lower for kw in vision_keywords):
            supports_vision = True
            logger.info(f"🔍 Auto-detected vision capability from model name: {active.get('model')}")
        
        # Build the prompt
        model_type = inference_data.get('processing_type', 'unknown')
        model_name = inference_data.get('model_name', 'unknown')
        
        # Create a clean version of inference data without large fields
        clean_data = {k: v for k, v in inference_data.items() 
                      if k not in ['result_image_base64', 'annotated_image', 'visualization']}
        
        system_prompt = """You are an ML inference results explainer. Your job is to provide clear, 
comprehensive explanations of machine learning model outputs to help users understand what the model found.

Be specific about:
1. What was detected/classified/segmented
2. The confidence levels and what they mean
3. How to interpret the visualization (colors, boxes, masks)
4. Any interesting observations or insights

Keep your explanation informative but concise (2-3 paragraphs max)."""

        # Build user message content
        if supports_vision and (image_base64 or result_image_base64):
            # Vision-capable model - send images
            content_parts = []
            
            # Add text prompt
            text_prompt = f"""Analyze these ML inference results and provide a detailed explanation for the user.

**Model**: {model_name}
**Type**: {model_type}

**Results Data**:
```json
{json.dumps(clean_data, indent=2)}
```

"""
            if image_base64 and result_image_base64:
                text_prompt += "I'm showing you the ORIGINAL image and the MODEL OUTPUT visualization. Compare them and explain what the model detected/processed."
            elif result_image_base64:
                text_prompt += "I'm showing you the MODEL OUTPUT visualization. Explain what the model found and what the colors/overlays represent."
            elif image_base64:
                text_prompt += "I'm showing you the ORIGINAL image. Based on the results data, explain what the model found in this image."
            
            content_parts.append({"type": "text", "text": text_prompt})
            
            # Add original image if available
            if image_base64:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}",
                        "detail": "low"  # Use low detail to save context
                    }
                })
            
            # Add result visualization if available
            if result_image_base64:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{result_image_base64}",
                        "detail": "low"
                    }
                })
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_parts}
            ]
        else:
            # Text-only model - just send the data
            user_prompt = f"""Analyze these ML inference results and provide a detailed explanation for the user.

**Model**: {model_name}
**Type**: {model_type}

**Results Data**:
```json
{json.dumps(clean_data, indent=2)}
```

Explain:
1. What the model does (based on the type)
2. What it found in the image (based on the results)
3. How to interpret the visualization that was generated
4. Any notable findings"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        
        # Make the LLM call with retry logic
        logger.info(f"🧠 Generating LLM explanation for {model_type} results (vision={supports_vision})")
        
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            try:
                response = router.chat(messages=messages, tools=None)
                
                if response and response.content:
                    logger.info(f"✅ LLM explanation generated ({len(response.content)} chars)")
                    return response.content
                else:
                    logger.warning(f"LLM returned empty response (attempt {attempt + 1}/{max_retries})")
            except Exception as retry_error:
                last_error = retry_error
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {retry_error}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Brief delay before retry
                    continue
        
        if last_error:
            logger.warning(f"All LLM retries failed: {last_error}")
        return None
            
    except Exception as e:
        logger.warning(f"Failed to generate LLM explanation: {e}")
        return None


def _load_image_as_base64(image_path: str, max_dimension: int = 512) -> Optional[str]:
    """
    Load and resize an image to base64 for LLM vision.
    Uses small dimensions to minimize context usage.
    """
    try:
        from PIL import Image
        import io
        
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # Resize to small dimensions for context efficiency
            width, height = img.size
            if max(width, height) > max_dimension:
                ratio = max_dimension / max(width, height)
                new_size = (int(width * ratio), int(height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=70)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        logger.warning(f"Failed to load image for LLM: {e}")
        return None


def run_inference(
    model_name: str,
    image_path: str,
    processing_type: str = 'auto',
    confidence_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Run inference on an image using a deployed model.
    
    This tool executes inference on the specified model with the uploaded image,
    automatically detecting or using the specified processing type to interpret results.
    
    Args:
        model_name: Name of the model to use for inference
        image_path: Path to the image file (from session storage)
        processing_type: Type of processing to apply ('auto', 'classification', 'detection', 
                        'segmentation', 'pose', 'keypoint', 'panoptic', 'ocr')
        confidence_threshold: Minimum confidence for detections (default: 0.5)
    
    Returns:
        Inference results with visualizations and interpretations
    """
    try:
        # Validate inputs
        if not model_name:
            return error_response(
                ValueError("model_name is required"),
                operation="run_inference"
            )
        
        if not image_path:
            return error_response(
                ValueError("image_path is required"),
                operation="run_inference"
            )
        
        # Security: Prevent path traversal attacks
        real_path = os.path.realpath(image_path)
        real_storage_root = os.path.realpath(SESSION_STORAGE_ROOT)
        if not real_path.startswith(real_storage_root + os.sep) and real_path != real_storage_root:
            return error_response(
                ValueError("Invalid file path - access denied"),
                operation="run_inference"
            )
        
        if not os.path.exists(image_path):
            return error_response(
                FileNotFoundError(f"Image not found: {image_path}"),
                operation="run_inference"
            )
        
        if processing_type not in AVAILABLE_PROCESSORS:
            return error_response(
                ValueError(f"Invalid processing_type: {processing_type}. Must be one of: {AVAILABLE_PROCESSORS}"),
                operation="run_inference"
            )
        
        # Validate confidence_threshold is within bounds
        if not 0.0 <= confidence_threshold <= 1.0:
            return error_response(
                ValueError("confidence_threshold must be between 0.0 and 1.0"),
                operation="run_inference"
            )
        
        # Import here to avoid circular imports
        from api.core import execute_prediction
        
        # Read image file
        with open(image_path, 'rb') as f:
            file_bytes = f.read()
        
        # Execute prediction
        result = execute_prediction(
            filepath=image_path,
            file_bytes=file_bytes,
            model_name=model_name,
            task_type=processing_type
        )
        
        if not result.get('success', False):
            error_msg = result.get('error', 'Inference failed')
            error_lower = error_msg.lower()
            
            # Build smart suggestions based on error type and what was already tried
            suggestions = []
            diagnostic_tools = []
            
            # Categorize the error
            is_model_error = "not ready" in error_lower or "not found" in error_lower
            is_input_error = "multi-input" in error_lower or "pixel_mask" in error_lower or "shape" in error_lower
            is_timeout_error = "timeout" in error_lower
            is_server_error = "server" in error_lower or "connection" in error_lower
            
            if is_model_error:
                suggestions.append("The model may still be loading or not deployed. Wait 10-15 seconds and retry.")
                diagnostic_tools.append("get_server_status")
                diagnostic_tools.append("list_available_models")
            
            if is_input_error:
                suggestions.append("This model has specific input requirements that may need special handling.")
                diagnostic_tools.append("get_model_config")
                diagnostic_tools.append("get_model_input_requirements")
            
            if is_timeout_error:
                suggestions.append("The request timed out. The model may be processing a large image or warming up.")
                suggestions.append("Try with a smaller image or wait for the model to warm up.")
            
            if is_server_error:
                suggestions.append("There may be a server connectivity issue.")
                diagnostic_tools.append("get_server_status")
            
            # Only suggest auto mode if not already using it
            if processing_type != "auto" and not is_input_error:
                suggestions.append(f"You used processing_type='{processing_type}'. Try processing_type='auto' for automatic detection.")
            
            # If auto was already used, suggest specific types based on model name
            if processing_type == "auto":
                model_lower = model_name.lower()
                if "detr" in model_lower or "yolo" in model_lower or "ssd" in model_lower or "rcnn" in model_lower:
                    suggestions.append("This appears to be a detection model. Try processing_type='detection' explicitly.")
                elif "resnet" in model_lower or "mobilenet" in model_lower or "efficientnet" in model_lower:
                    suggestions.append("This may be a classification model. Try processing_type='classification' explicitly.")
                elif "segment" in model_lower or "mask" in model_lower:
                    suggestions.append("This may be a segmentation model. Try processing_type='segmentation' explicitly.")
                else:
                    # Generic fallback - don't suggest auto again
                    suggestions.append("Auto-detection failed. Check model config to determine the correct processing type.")
            
            # Add diagnostic tool suggestions
            if diagnostic_tools:
                unique_tools = list(dict.fromkeys(diagnostic_tools))  # Remove duplicates, preserve order
                suggestions.append(f"Run these diagnostic tools first: {', '.join(unique_tools)}")
            
            # Ensure we always have at least one suggestion
            if not suggestions:
                suggestions.append("Check model configuration with get_model_config and server status with get_server_status.")
            
            return {
                "success": False,
                "error": error_msg,
                "model_name": model_name,
                "processing_type_used": processing_type,
                "suggestions": suggestions,
                "recommended_diagnostics": diagnostic_tools if diagnostic_tools else None,
                "context": f"Inference failed for model {model_name} with processing_type='{processing_type}'"
            }
        
        # Build response with key information
        inference_result = {
            "success": True,
            "model_name": model_name,
            "processing_type": result.get('detected_type', processing_type),
            "auto_detected": result.get('auto_detected', False),
            "image_path": image_path,
        }
        
        # Add timing information – expose granular latency breakdown
        latency_info: Dict[str, Any] = {}
        if 'inference_time' in result:
            inference_ms = round(result['inference_time'] * 1000, 2)
            inference_result["inference_time_ms"] = inference_ms
            latency_info["inference_ms"] = inference_ms
        if 'total_time' in result:
            latency_info["total_ms"] = round(result['total_time'] * 1000, 2)
        if 'timing' in result and isinstance(result['timing'], dict):
            for k, v in result['timing'].items():
                if k not in latency_info and isinstance(v, (int, float)):
                    latency_info[k] = round(v, 3)
        if latency_info:
            inference_result["latency"] = latency_info
        
        # Add results based on processing type
        detected_type = result.get('detected_type', processing_type)
        
        if detected_type == 'classification':
            top_preds = result.get('top_predictions', result.get('predictions', []))[:5]
            inference_result["predictions"] = top_preds
            if top_preds:
                top = top_preds[0]
                top_class = top.get('class_name', top.get('class', 'unknown'))
                top_conf = top.get('confidence', 0)
                inference_result["summary"] = f"Top prediction: {top_class} ({top_conf:.1%} confidence)"
                # Build full list for agent
                pred_lines = []
                for p in top_preds:
                    pname = p.get('class_name', p.get('class', 'unknown'))
                    pconf = p.get('confidence', 0)
                    pred_lines.append(f"  - {pname}: {pconf:.1%}")
                all_preds_str = "\n".join(pred_lines)
                inference_result["explanation"] = (
                    f"The classification model analyzed the image and returned these top predictions:\n"
                    f"{all_preds_str}\n"
                    f"The highest-confidence class is '{top_class}' at {top_conf:.1%}. "
                    f"Classification models assign the entire image to a single category from a set of predefined classes. "
                    f"Note: confidence values represent relative probabilities across all classes."
                )
        
        elif detected_type == 'detection':
            detections = result.get('detections', [])
            # Filter by confidence
            filtered = [d for d in detections if d.get('confidence', 0) >= confidence_threshold]
            inference_result["detections"] = filtered[:20]  # Limit to 20
            inference_result["total_detections"] = len(detections)
            inference_result["filtered_detections"] = len(filtered)
            
            # Summarize by class (detection.py uses 'class_name' key)
            class_counts = {}
            for d in filtered:
                cls = d.get('class_name', d.get('class', 'unknown'))
                class_counts[cls] = class_counts.get(cls, 0) + 1
            inference_result["class_summary"] = class_counts
            inference_result["summary"] = f"Detected {len(filtered)} objects: " + ", ".join(
                f"{count} {cls}" for cls, count in sorted(class_counts.items(), key=lambda x: -x[1])[:5]
            )
            inference_result["explanation"] = (
                f"The detection model found and localized {len(filtered)} objects in the image "
                f"(after filtering with confidence threshold {confidence_threshold}). "
                f"Each detection includes a bounding box showing where the object is located. "
                f"Objects found: {', '.join(f'{v} {k}(s)' for k, v in class_counts.items())}."
            )
        
        elif detected_type == 'segmentation':
            inference_result["classes_found"] = result.get('class_stats', [])
            inference_result["num_classes"] = result.get('num_classes', 0)
            inference_result["mask_shape"] = result.get('mask_shape', [])
            # Include top classes in summary
            class_stats = result.get('class_stats', [])
            top_class_parts = []
            class_details = []
            for c in class_stats[:5]:
                class_name = c.get('class_name') or f"Class_{c.get('class_id')}"
                percentage = c.get('percentage', 0)
                top_class_parts.append(f"{class_name} ({percentage:.1f}%)")
                class_details.append(f"'{class_name}' covering {percentage:.1f}% of the image")
            top_classes = ", ".join(top_class_parts)
            inference_result["summary"] = f"Segmentation found {result.get('num_classes', 0)} classes. Top classes: {top_classes}"
            inference_result["explanation"] = (
                f"The segmentation model classified every pixel in the image into one of {result.get('num_classes', 0)} categories. "
                f"Unlike classification (which labels the whole image) or detection (which draws boxes), "
                f"segmentation creates a detailed mask showing exactly which pixels belong to each class. "
                f"The visualization shows each class in a different color. "
                f"Classes found: {'; '.join(class_details) if class_details else 'see class_stats for details'}."
            )
        
        elif detected_type == 'pose':
            # pose.py returns 'num_poses' and 'poses' list
            num_poses = result.get('num_poses', 0)
            poses = result.get('poses', [])
            inference_result["num_people"] = num_poses
            inference_result["poses"] = poses
            if poses:
                kp_count = poses[0].get('num_keypoints', 0)
                inference_result["keypoints_per_person"] = kp_count
                # Build keypoint summary for the agent
                pose_details = []
                for p in poses:
                    pid = p.get('person_id', 0)
                    pconf = p.get('confidence', 0)
                    pose_details.append(f"Person {pid}: confidence {pconf:.1%}, {p.get('num_keypoints', 0)} keypoints")
                poses_str = "; ".join(pose_details)
                inference_result["summary"] = f"Detected {num_poses} person(s) with pose estimation. {poses_str}"
                inference_result["explanation"] = (
                    f"The pose estimation model detected {num_poses} person(s) in the image "
                    f"and identified {kp_count} body keypoints for each person. "
                    f"Details: {poses_str}. "
                    f"These keypoints typically include joints like shoulders, elbows, wrists, hips, knees, and ankles, "
                    f"as well as facial landmarks. The visualization connects these points to show the body pose."
                )
            else:
                inference_result["summary"] = "Pose estimation completed but detected 0 people"
                inference_result["explanation"] = (
                    "The pose estimation model did not find any human figures above the confidence threshold. "
                    "This may happen if people are occluded, too small, or in unusual poses."
                )
        
        elif detected_type == 'keypoint':
            # keypoint.py returns 'num_instances' and 'keypoint_results'
            num_instances = result.get('num_instances', 0)
            keypoint_results = result.get('keypoint_results', [])
            inference_result["num_people"] = num_instances
            inference_result["keypoint_results"] = keypoint_results
            inference_result["summary"] = f"Detected {num_instances} instance(s) with keypoint detection"
            inference_result["explanation"] = (
                f"The keypoint detection model found {num_instances} instance(s) in the image "
                f"and identified body keypoints for each. "
                f"These keypoints typically include joints like shoulders, elbows, wrists, hips, knees, and ankles, "
                f"as well as facial landmarks. The visualization connects these points to show the body pose."
            )
        
        elif detected_type == 'panoptic':
            # panoptic.py returns 'num_segments' and 'segments'
            num_segments = result.get('num_segments', 0)
            segments = result.get('segments', [])
            inference_result["num_segments"] = num_segments
            inference_result["segments"] = segments[:20]
            # Summarize segments
            seg_details = []
            for s in segments[:10]:
                seg_name = s.get('class_name', s.get('label', f"Segment_{s.get('id', '?')}"))
                seg_details.append(seg_name)
            seg_str = ", ".join(seg_details) if seg_details else "none"
            inference_result["summary"] = f"Panoptic segmentation found {num_segments} segments: {seg_str}"
            inference_result["explanation"] = (
                f"The panoptic segmentation model identified {num_segments} distinct segments in the image. "
                f"Panoptic segmentation combines instance segmentation (individual objects) and semantic segmentation "
                f"(pixel-level classification), giving each object and background region a unique identity. "
                f"Segments found: {seg_str}."
            )
        
        elif detected_type == 'ocr':
            # ocr.py returns 'recognized_text', not 'text'
            ocr_text = result.get('recognized_text', result.get('text', ''))
            ocr_confidence = result.get('confidence', 0)
            inference_result["text"] = ocr_text
            inference_result["confidence"] = ocr_confidence
            text_preview = ocr_text[:100] if ocr_text else ''
            inference_result["summary"] = f"OCR result: '{text_preview}'" if ocr_text else "No text detected"
            ocr_found = f'extracted: "{text_preview}"' if ocr_text else 'found no readable text'
            inference_result["explanation"] = (
                f"The OCR (Optical Character Recognition) model scanned the image for text and "
                f"{ocr_found}. "
                f"OCR models convert images of text into machine-readable strings."
            )
        
        else:
            inference_result["raw_output_shapes"] = result.get('output_shapes', [])
            inference_result["summary"] = f"Inference completed with output shapes: {result.get('output_shapes', [])}"
            inference_result["explanation"] = (
                f"The model produced raw output tensors with shapes: {result.get('output_shapes', [])}. "
                f"The specific interpretation depends on what the model was trained to do. "
                f"Consider using analyze_model_type tool to understand this model better."
            )
        
        # Add visualization (base64 image) if available
        # Check various keys that might contain the result image
        result_image_b64 = (
            result.get('annotated_image') or  # segmentation, detection
            result.get('result_image') or     # generic
            result.get('visualization') or    # alternative key
            result.get('output_image')        # another alternative
        )
        
        if result_image_b64:
            inference_result["visualization_available"] = True
            inference_result["result_image_base64"] = result_image_b64
            inference_result["has_visualization"] = True
            
            # Save the visualization to a file so view_image tool can access it
            # Extract session directory from the image_path
            session_dir = os.path.dirname(image_path)
            result_image_filename = f"result_{model_name}_{detected_type}.png"
            result_image_path = os.path.join(session_dir, result_image_filename)
            
            try:
                # Decode base64 and save to file
                import base64
                image_data = base64.b64decode(result_image_b64)
                with open(result_image_path, 'wb') as f:
                    f.write(image_data)
                inference_result["result_image_path"] = result_image_path
                logger.info(f"✅ Saved result visualization to {result_image_path}")
                
                inference_result["visualization_description"] = (
                    f"A visualization image is available showing the {detected_type} results "
                    f"overlaid on the original image. Use view_image tool with path: {result_image_path}"
                )
            except Exception as e:
                logger.warning(f"Failed to save result image: {e}")
                inference_result["visualization_description"] = (
                    f"A visualization image is available showing the {detected_type} results "
                    f"overlaid on the original image. The image is in result_image_base64."
                )
        else:
            inference_result["visualization_available"] = False
            inference_result["has_visualization"] = False
        
        # Generate rich LLM explanation using a dedicated LLM instance
        # This keeps the main agent context clean while providing detailed analysis
        try:
            # Load original image for LLM vision (small size to save context)
            original_image_b64 = _load_image_as_base64(image_path, max_dimension=512)
            
            # Use smaller version of result image for LLM
            result_image_for_llm = None
            if result_image_b64:
                # The result image is already base64, but we might want to resize it
                # For now, just use it directly (it should already be reasonably sized)
                result_image_for_llm = result_image_b64
            
            # Generate LLM explanation
            llm_explanation = _generate_llm_explanation(
                inference_data=inference_result,
                image_base64=original_image_b64,
                result_image_base64=result_image_for_llm
            )
            
            if llm_explanation:
                inference_result["llm_analysis"] = llm_explanation
                inference_result["analysis_source"] = "llm"
                logger.info("✅ Added LLM-generated analysis to inference results")
            else:
                inference_result["analysis_source"] = "template"
                logger.info("ℹ️ Using template-based explanation (LLM unavailable)")
                
        except Exception as e:
            logger.warning(f"LLM explanation generation failed: {e}")
            inference_result["analysis_source"] = "template"
        
        return ok(
            data=inference_result,
            message=inference_result.get("summary", "Inference completed successfully")
        )
        
    except Exception as e:
        logger.error(f"Error running inference: {e}", exc_info=True)
        return error_response(
            e,
            operation="run_inference",
            model_name=model_name,
            image_path=image_path
        )


def list_processing_types() -> Dict[str, Any]:
    """
    List available processing types for inference.
    
    Returns information about each processing type to help
    choose the right one for a given model.
    """
    processing_info = {
        "auto": {
            "description": "Automatically detect model type from name and output shapes",
            "use_when": "You're unsure about the model type",
            "confidence": "Medium - based on heuristics"
        },
        "classification": {
            "description": "Image classification - assigns class labels to entire image",
            "use_when": "Model outputs class probabilities (e.g., ResNet, MobileNet, EfficientNet)",
            "typical_outputs": "[batch, num_classes]"
        },
        "detection": {
            "description": "Object detection - finds and localizes objects with bounding boxes",
            "use_when": "Model outputs bounding boxes (e.g., YOLO, SSD, Faster R-CNN)",
            "typical_outputs": "[batch, num_boxes, 5+num_classes] or similar"
        },
        "segmentation": {
            "description": "Semantic segmentation - classifies each pixel",
            "use_when": "Model outputs per-pixel class masks (e.g., DeepLab, U-Net)",
            "typical_outputs": "[batch, num_classes, height, width]"
        },
        "pose": {
            "description": "Pose estimation - detects human body keypoints",
            "use_when": "Model outputs keypoint coordinates (e.g., OpenPose, HRNet)",
            "typical_outputs": "[batch, num_people, num_keypoints, 2-3]"
        },
        "keypoint": {
            "description": "General keypoint detection - similar to pose but for any keypoints",
            "use_when": "Model outputs keypoint heatmaps or coordinates",
            "typical_outputs": "[batch, num_keypoints, height, width] for heatmaps"
        },
        "panoptic": {
            "description": "Panoptic segmentation - combines instance and semantic segmentation",
            "use_when": "Model outputs both instance masks and semantic classes",
            "typical_outputs": "Multiple outputs for instances and semantics"
        },
        "ocr": {
            "description": "Optical Character Recognition - extracts text from images",
            "use_when": "Model outputs text sequences (e.g., CRNN, TrOCR)",
            "typical_outputs": "[batch, sequence_length, vocab_size]"
        }
    }
    
    return ok(
        data={
            "processing_types": processing_info,
            "available_types": AVAILABLE_PROCESSORS
        },
        message="Available processing types for inference"
    )


# Register the tools
register_tool(
    name="run_inference",
    func=run_inference,
    description="Run ML inference on an uploaded image. Returns results AND a visualization image showing what the model found. IMPORTANT: Read the 'explanation' field in the result to understand and explain to the user what the model did. For segmentation, detection, pose, etc., the visualization shows colored regions, bounding boxes, or keypoints overlaid on the image.",
    input_schema={
        "type": "object",
        "properties": {
            "model_name": {
                "type": "string",
                "description": "Name of the model to use for inference"
            },
            "image_path": {
                "type": "string", 
                "description": "Path to the uploaded image file"
            },
            "processing_type": {
                "type": "string",
                "enum": AVAILABLE_PROCESSORS,
                "default": "auto",
                "description": "Processing type: 'auto' to auto-detect, or specify 'classification', 'detection', 'segmentation', 'pose', 'keypoint', 'panoptic', or 'ocr'"
            },
            "confidence_threshold": {
                "type": "number",
                "default": 0.5,
                "description": "Minimum confidence threshold for detections (0.0-1.0)"
            }
        },
        "required": ["model_name", "image_path"]
    }
)

register_tool(
    name="list_processing_types",
    func=list_processing_types,
    description="List available processing types for inference with descriptions. Use this to help users understand which processing type to use for their model.",
    input_schema={
        "type": "object",
        "properties": {},
        "required": []
    }
)
