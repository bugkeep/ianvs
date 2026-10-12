# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import copy
import uuid
from datetime import datetime, timezone


class EventBus:
    def __init__(self, store, session_id=None, instance_id=None, task_id=None):
        self.store = store
        self.identities = {"session_id": session_id, "instance_id": instance_id, "task_id": task_id, "environment_id": None, "run_id": None}
        self.subscribers = []

    def bind(self, **identities):
        self.identities.update(identities)

    def subscribe(self, callback):
        self.subscribers.append(callback)
        return lambda: self.subscribers.remove(callback)

    def emit(self, event_type, payload, **identities):
        event = {
            "schema_version": 1,
            "event_id": uuid.uuid4().hex,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **self.identities,
            **identities,
            "event_type": event_type,
            "payload": copy.deepcopy(payload),
        }
        self.store.append_sequenced_event("events.jsonl", event)
        if event.get("session_id") and event.get("instance_id"):
            relative = f"instances/{event['instance_id']}/sessions/{event['session_id']}/events.jsonl"
            self.store.append(relative, event)
        for callback in tuple(self.subscribers):
            callback(event)
        return event
