import json
from typing import Any, Dict

class EventBus:
    """
    Real-time event bus for pushing telemetry to the Android UI.
    In a real implementation, this would write to a socket, pipe, or message queue.
    """

    @staticmethod
    def emit(event_type: str, data: Dict[str, Any]):
        payload = {
            "type": event_type,
            "data": data,
            "timestamp": float
        }
        # For this prototype, we print to stdout which the Worker service captures
        print(f"EVENT_BUS_EMIT: {json.dumps(payload)}")

    @staticmethod
    def emit_step_progress(step_id: str, status: str, message: str):
        EventBus.emit("STEP_PROGRESS", {
            "step_id": step_id,
            "status": status,
            "message": message
        })
