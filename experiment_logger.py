import shelve
import json
import inspect
import time
import uuid
import functools
from pathlib import Path
import contextlib
from typing import Any, Dict, Optional, Iterator, Callable, TypeVar, Union, cast, List
import traceback
import pandas as pd
import dill

T = TypeVar('T', bound=Callable[..., Any])

shelve.Pickler = dill.Pickler
shelve.Unpickler = dill.Unpickler

class ExperimentLogger:
    """A stateful logger that links logs within function calls using trace IDs."""
    
    def __init__(self, 
                 log_directory: Union[str, Path] = 'experiment_logs', 
                 **metadata: Any) -> None:
        """Initialize the logger with a directory and base metadata."""
        self.log_dir = Path(log_directory)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        self.objects_path = str(self.log_dir / 'objects')
        self.metadata_path = self.log_dir / 'metadata.jsonl'
        self.base_metadata = metadata
        
        # Stack for trace IDs (supports nested function calls)
        self._trace_stack: list[str] = []
        
        if not self.metadata_path.exists():
            self.metadata_path.touch()

    @contextlib.contextmanager
    def _trace_context(self, trace_id: str):
        """Context manager for tracking the current function's trace ID."""
        self._trace_stack.append(trace_id)
        try:
            yield
        finally:
            self._trace_stack.pop()
    
    def _get_caller_info(self) -> Dict[str, Any]:
        """Get information about the actual calling function."""
        frame = inspect.currentframe()
        if frame is None:
            return {}
            
        caller = frame.f_back
        if caller is None:
            return {}
            
        # Skip logger internal frames and decorators
        while caller:
            code = caller.f_code
            if (code.co_filename == __file__ and 
                code.co_name in ['log', '_trace_context', '_decorator', 'wrapper']):
                caller = caller.f_back
            else:
                break
                
        if caller is None:
            return {}
            
        return {
            'function_name': caller.f_code.co_name,
        }

    def _get_variable_name(self, obj: Any, frame_locals: Dict[str, Any]) -> Optional[str]:
        """Get the variable name for an object in the given frame locals."""
        try:
            obj_id = id(obj)
            names = [
                name for name, value in reversed(list(frame_locals.items())) 
                if id(value) == obj_id
            ]
            return names[0] if names else None
        except Exception:
            return None

    def log(self, obj: Any, **metadata: Any) -> None:
        """Log an object with metadata, trace stack, and variable name."""
        try:
            timestamp = time.time()
            obj_id = f"{timestamp:.7f}"
            
            # Get caller information
            caller_info = self._get_caller_info()
            
            # Get variable name
            if frame := inspect.currentframe():
                if caller := frame.f_back:
                    if var_name := self._get_variable_name(obj, caller.f_locals):
                        caller_info['variable_name'] = var_name
            
            # Build metadata
            combined_metadata = {
                'id': obj_id,
                **caller_info,
                **self.base_metadata,
            }
            
            # Add trace stack if available
            if self._trace_stack:
                combined_metadata['trace_id'] = self._trace_stack[-1]  # Current trace ID
                combined_metadata['trace_stack'] = self._trace_stack.copy()  # Full stack
            
            # Add call-specific metadata
            combined_metadata.update(metadata)
            
            # Convert any tensor values in metadata to Python native types
            # Try to import torch if available (optional dependency)
            try:
                import torch
                has_torch = True
            except ImportError:
                has_torch = False
            
            def convert_tensors(obj):
                """Recursively convert tensors to Python native types."""
                if has_torch and isinstance(obj, torch.Tensor):
                    if obj.numel() == 1:
                        return obj.item()
                    else:
                        return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_tensors(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_tensors(item) for item in obj]
                else:
                    return obj
            
            combined_metadata = convert_tensors(combined_metadata)
            
            # Handle large objects (like attention tensors) by saving to disk separately
            # Check if object is a large tensor or tuple of tensors
            is_large_tensor = False
            if has_torch:
                if isinstance(obj, torch.Tensor) and obj.numel() > 1000000:  # > 1M elements
                    is_large_tensor = True
                elif isinstance(obj, (tuple, list)) and len(obj) > 0:
                    # Check if it's a tuple/list of tensors (like attention outputs)
                    if isinstance(obj[0], torch.Tensor) and sum(t.numel() for t in obj if isinstance(t, torch.Tensor)) > 1000000:
                        is_large_tensor = True
            
            if is_large_tensor:
                # Save large tensors as separate .pt files
                objects_dir = self.log_dir / 'tensor_objects'
                objects_dir.mkdir(exist_ok=True)
                tensor_file_path = objects_dir / f"{obj_id}.pt"
                torch.save(obj, tensor_file_path)
                combined_metadata['tensor_file'] = str(tensor_file_path.relative_to(self.log_dir))
                combined_metadata['object_type'] = 'large_tensor'
                # Don't store in shelve, just metadata
            else:
                # Store small objects in shelve
                with shelve.open(self.objects_path) as shelf:
                    shelf[obj_id] = obj
            
            # Write metadata
            with open(self.metadata_path, 'a', encoding='utf-8') as f:
                json.dump(combined_metadata, f)
                f.write('\n')
                
        except Exception as e:
            raise RuntimeError(f"Error logging object: {str(e)}") from e
    
    def query(self, metadata_query: Dict[str, Any]) -> Iterator[Any]:
        """
        Query logged objects based on metadata.
        Special handling for trace_id to match anywhere in the trace stack.
        """
        def matches_query(metadata: Dict[str, Any], query: Dict[str, Any]) -> bool:
            for k, v in query.items():
                if k == 'trace_id':
                    direct_match = metadata.get('trace_id') == v
                    stack_match = v in metadata.get('trace_stack', []) 
                    if not (direct_match or stack_match):
                        return False
                else:
                    # Regular metadata matching
                    if metadata.get(k) != v:
                        return False
            return True

        try:
            with shelve.open(self.objects_path) as shelf:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            metadata = json.loads(line.strip())
                            if matches_query(metadata, metadata_query):
                                obj_id = metadata['id']
                                
                                # Check if object is stored as tensor file
                                if 'tensor_file' in metadata:
                                    tensor_path = self.log_dir / metadata['tensor_file']
                                    if tensor_path.exists():
                                        try:
                                            import torch
                                            yield torch.load(tensor_path, map_location='cpu')
                                        except Exception as e:
                                            raise RuntimeError(f"Error loading tensor file {tensor_path}: {str(e)}") from e
                                elif shelf.get(obj_id, None) is not None:
                                    yield shelf.get(obj_id)
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            raise RuntimeError(f"Error querying objects: {str(e)}") from e

