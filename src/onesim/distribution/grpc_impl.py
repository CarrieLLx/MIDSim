import asyncio
import grpc
import json
from typing import Dict, Any, Optional, List, Tuple
from concurrent import futures
import threading
import uuid
import time
from loguru import logger
import os
import sys
from pathlib import Path
from datetime import datetime
import weakref

# Import ConnectionManager from the new module
from onesim.distribution.connection_manager import connection_manager
# Import BatchProcessor from the new module
from onesim.distribution.batch_processor import batch_processor

# Get the current directory.
current_dir = Path(__file__).parent.absolute()

# Add the current directory to sys.path first so modules can be imported correctly.
sys.path.insert(0, str(current_dir))

# Try to import the proto modules.
try:
    # Try direct imports without a package path.
    import onesim.distribution.agent_proto.agent_pb2 as agent_pb2
    import onesim.distribution.agent_proto.agent_pb2_grpc as agent_pb2_grpc
    
    logger.info("Successfully imported proto modules")
except ImportError as e:
    logger.warning(f"Failed to import proto modules: {e}")
    logger.info("Generating protobuf files...")
    
    # Clean the existing proto directory to resolve possible nested-directory issues.
    import shutil
    proto_dir = current_dir / "agent_proto"
    if proto_dir.exists():
        logger.info(f"Removing existing proto directory: {proto_dir}")
        shutil.rmtree(proto_dir)
    
    # Create a new proto directory.
    proto_dir.mkdir(exist_ok=True)
    
    # Create the proto file.
    proto_file = proto_dir / "agent.proto"
    with open(proto_file, "w") as f:
        f.write("""
syntax = "proto3";

package agent;

// Event service definition.
service AgentService {
  // Register a worker node.
  rpc RegisterWorker (RegisterWorkerRequest) returns (RegisterWorkerResponse) {}
  
  // Heartbeat service.
  rpc Heartbeat (HeartbeatRequest) returns (HeartbeatResponse) {}
  
  // Create an agent on a worker.
  rpc CreateAgent (CreateAgentRequest) returns (CreateAgentResponse) {}
  
  // Send an event.
  rpc SendEvent (EventRequest) returns (EventResponse) {}
  
  // Create agents in a batch.
  rpc CreateAgentsBatch (CreateAgentsBatchRequest) returns (CreateAgentsBatchResponse) {}
  
  // Send a data storage event.
  rpc SendStorageEvent (StorageEventRequest) returns (StorageEventResponse) {}
  
  // Send data storage events in a batch.
  rpc SendStorageEventBatch (StorageEventBatchRequest) returns (StorageEventBatchResponse) {}
  
  // Send a decision record.
  rpc SendDecisionRecord (DecisionRecordRequest) returns (DecisionRecordResponse) {}
  
  // Send decision records in a batch.
  rpc SendDecisionRecordBatch (DecisionRecordBatchRequest) returns (DecisionRecordBatchResponse) {}
  
  // Get environment data.
  rpc GetEnvData (EnvDataRequest) returns (EnvDataResponse) {}
  
  // Update environment data.
  rpc UpdateEnvData (EnvDataUpdateRequest) returns (EnvDataUpdateResponse) {}
  
  // Request simulation termination.
  rpc StopSimulation (SimulationStopRequest) returns (SimulationStopResponse) {}
  
  // Get agent data.
  rpc GetAgentData (AgentDataRequest) returns (AgentDataResponse) {}
  
  // Get data for agents of a specified type.
  rpc GetAgentDataByType (AgentDataByTypeRequest) returns (AgentDataByTypeResponse) {}
  
  // Get token usage for a node.
  rpc GetTokenUsage (TokenUsageRequest) returns (TokenUsageResponse) {}

  // Locate the worker hosting an agent.
  rpc LocateAgent (LocateAgentRequest) returns (LocateAgentResponse) {}

  // Collect data from workers in a batch at the master's request.
  rpc CollectDataBatch(BatchDataRequest) returns (BatchDataResponse) {}
}

// Worker registration request.
message RegisterWorkerRequest {
  string worker_id = 1;
  string address = 2;
  int32 port = 3;
}

// Worker registration response.
message RegisterWorkerResponse {
  bool success = 1;
  string message = 2;
}

// Heartbeat request.
message HeartbeatRequest {
  string worker_id = 1;
  int64 timestamp = 2;
}

// Heartbeat response.
message HeartbeatResponse {
  bool acknowledged = 1;
}

// Agent creation request.
message CreateAgentRequest {
  string agent_type = 1;
  string agent_id = 2;
  string config_json = 3;  // JSON-serialized configuration.
}

// Agent creation response.
message CreateAgentResponse {
  bool success = 1;
  string message = 2;
  string agent_id = 3;
}

// Event request.
message EventRequest {
  string event_id = 1;
  string event_kind = 2;
  string from_agent_id = 3;
  string to_agent_id = 4;
  int64 timestamp = 5;
  string payload_json = 6;  // JSON-serialized payload.
  string reply_to_worker_address = 7; // For P2P direct replies
  int32 reply_to_worker_port = 8;    // For P2P direct replies
}

// Event response.
message EventResponse {
  bool received = 1;
}

// Batch event request for P2P delivery.
message EventBatchRequest {
  repeated EventRequest events = 1;
}

// Batch event response.
message EventBatchResponse {
  bool received = 1;
  int32 processed_count = 2;
  string error = 3;
}

// Data storage event request.
message StorageEventRequest {
  string event_type = 1;
  string source_type = 2;
  string source_id = 3;
  string target_type = 4;
  string target_id = 5;
  string payload_json = 6;  // JSON-serialized payload.
}

// Data storage event response.
message StorageEventResponse {
  bool received = 1;
}

// Batch data storage event request.
message StorageEventBatchRequest {
  repeated StorageEventRequest events = 1;
}

// Batch data storage event response.
message StorageEventBatchResponse {
  bool received = 1;
  int32 processed_count = 2;
  string error = 3;
}

// Decision record request.
message DecisionRecordRequest {
  string agent_id = 1;
  string prompt = 2;
  string output = 3;
  double processing_time = 4;
  string context_json = 5;  // JSON-serialized context.
}

// Decision record response.
message DecisionRecordResponse {
  bool received = 1;
}

// Batch decision record request.
message DecisionRecordBatchRequest {
  repeated DecisionRecordRequest decisions = 1;
}

// Batch decision record response.
message DecisionRecordBatchResponse {
  bool received = 1;
  int32 processed_count = 2;
  string error = 3;
}

// Batch agent creation request.
message CreateAgentsBatchRequest {
  repeated string configs_json = 1;
}

// Batch agent creation response.
message CreateAgentsBatchResponse {
  bool success = 1;
  string message = 2;
  repeated string agent_ids = 3;
}

// Environment data request.
message EnvDataRequest {
  string key = 1;
  string default_value_json = 2;  // JSON-serialized default value.
}

// Environment data response.
message EnvDataResponse {
  bool success = 1;
  string value_json = 2;  // JSON-serialized value.
  string error = 3;
}

// Environment data update request.
message EnvDataUpdateRequest {
  string key = 1;
  string value_json = 2;  // JSON-serialized value.
}

// Environment data update response.
message EnvDataUpdateResponse {
  bool success = 1;
  string error = 2;
}

// Simulation termination request.
message SimulationStopRequest {
  string worker_id = 1;
  string reason = 2;
  int64 timestamp = 3;
}

// Simulation termination response.
message SimulationStopResponse {
  bool acknowledged = 1;
  string message = 2;
}

// Agent data request.
message AgentDataRequest {
  string agent_id = 1;
  string key = 2;
  string default_value_json = 3;  // JSON-serialized default value.
}

// Agent data response.
message AgentDataResponse {
  bool success = 1;
  string value_json = 2;  // JSON-serialized value.
  string error = 3;
}

// Request for agent data by type.
message AgentDataByTypeRequest {
  string agent_type = 1;
  string key = 2;
  string default_value_json = 3;  // JSON-serialized default value.
}

// Response containing agent data by type.
message AgentDataByTypeResponse {
  bool success = 1;
  string values_json = 2;  // JSON-serialized {agent_id: value} dictionary.
  string error = 3;
}

// Agent location request.
message LocateAgentRequest {
  string agent_id = 1;
}

// Agent location response.
message LocateAgentResponse {
  bool success = 1;
  string worker_address = 2;
  int32 worker_port = 3;
  string error_message = 4;
}

// Token usage request.
message TokenUsageRequest {
  string worker_id = 1;
}

// Token usage response.
message TokenUsageResponse {
  bool success = 1;
  string token_stats_json = 2;  // JSON-serialized token statistics.
  string error = 3;
}

// Master request to collect data from workers in a batch.
message BatchDataRequest {
  string agent_type = 1;          // Agent type to query.
  string data_key = 2;            // Key of the requested data.
  string default_value_json = 3;  // Optional JSON-serialized default value.
}

// Response to the master's batch data collection request.
message BatchDataResponse {
  bool success = 1;
  string collected_data_json = 2; // JSON-serialized collected data: {agent_id: value}.
  string error_message = 3;       // Error message when success is false.
}        """)
    
    # Create __init__.py to make proto a package.
    init_file = proto_dir / "__init__.py"
    with open(init_file, "w") as f:
        f.write("# Proto package\n")
    
    # Run protoc to generate Python code.
    import subprocess
    try:
        # Log the full command for debugging.
        cmd = [
        "python", "-m", "grpc_tools.protoc",
        f"--proto_path={str(current_dir)}",
        f"--python_out={str(current_dir)}",
        f"--grpc_python_out={str(current_dir)}",
        "agent_proto/agent.proto"  # Use a path relative to proto_path, not str(proto_file).
    ]
        logger.info(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd, 
            check=True,
            capture_output=True,
            text=True
        )
        
        logger.info(f"Command output: {result.stdout}")
        if result.stderr:
            logger.warning(f"Command stderr: {result.stderr}")
            
        logger.info("Successfully generated protobuf files")
        
        # Import the generated modules again.
        import onesim.distribution.agent_proto.agent_pb2 as agent_pb2
        import onesim.distribution.agent_proto.agent_pb2_grpc as agent_pb2_grpc
    except Exception as e:
        logger.error(f"Failed to generate protobuf files: {e}")
        raise ImportError(f"Could not generate or import protobuf files: {e}")

def event_to_proto(event) -> agent_pb2.EventRequest:
    """Convert an event object to protobuf message."""
    from onesim.events import Event, DataEvent, DataResponseEvent, DataUpdateEvent, DataUpdateResponseEvent

    # Use the event's existing ID and timestamp if available
    event_id = getattr(event, 'event_id', str(uuid.uuid4()))
    # Convert timestamp to milliseconds if it's in seconds
    if hasattr(event, 'timestamp'):
        timestamp = int(event.timestamp * 1000)
    else:
        timestamp = int(time.time() * 1000)
    
    # Create payload dictionary
    payload = {}
    
    # Add basic properties to payload
    for key, value in vars(event).items():
        if key not in ['from_agent_id', 'to_agent_id', 'event_kind', 'event_id', 'timestamp']:
            payload[key] = value
    
    # Special handling for different event types
    if isinstance(event, DataEvent):
        payload['source_type'] = event.source_type
        payload['target_type'] = event.target_type
        payload['key'] = event.key
        if hasattr(event, 'default') and event.default is not None:
            payload['default'] = event.default
    
    elif isinstance(event, DataResponseEvent):
        payload['key'] = event.key
        payload['success'] = event.success
        if hasattr(event, 'data_value') and event.data_value is not None:
            payload['data_value'] = event.data_value
        if hasattr(event, 'error') and event.error is not None:
            payload['error'] = event.error
    
    elif isinstance(event, DataUpdateEvent):
        payload['source_type'] = event.source_type
        payload['target_type'] = event.target_type
        payload['key'] = event.key
        payload['value'] = event.value
    
    elif isinstance(event, DataUpdateResponseEvent):
        payload['key'] = event.key
        payload['success'] = event.success
        if hasattr(event, 'error') and event.error is not None:
            payload['error'] = event.error
            
    # Convert payload to JSON
    payload_json = json.dumps(payload, default=lambda o: str(o))
    
    # Create protobuf message
    return agent_pb2.EventRequest(
        event_id=event_id,
        event_kind=event.event_kind,
        from_agent_id=event.from_agent_id,
        to_agent_id=event.to_agent_id,
        timestamp=timestamp,
        payload_json=payload_json
    )

def proto_to_event(proto_event) -> Any:
    """Convert a protobuf message to an event object."""
    from onesim.events import Event, DataEvent, DataResponseEvent, DataUpdateEvent, DataUpdateResponseEvent, EndEvent
    
    # Parse payload from JSON
    try:
        payload = json.loads(proto_event.payload_json)
    except json.JSONDecodeError:
        payload = {}
    
    # Convert timestamp from milliseconds to seconds
    timestamp = proto_event.timestamp / 1000.0
    
    # Common parameters for all event types
    common_params = {
        'event_id': proto_event.event_id,
        'timestamp': timestamp
    }
    
    # Determine the event type and create appropriate object
    event_kind = proto_event.event_kind
    
    if event_kind == 'DataEvent':
        event = DataEvent(
            from_agent_id=proto_event.from_agent_id,
            to_agent_id=proto_event.to_agent_id,
            source_type=payload.get('source_type', 'UNKNOWN'),
            target_type=payload.get('target_type', 'UNKNOWN'),
            key=payload.get('key', ''),
            default=payload.get('default'),
            request_id=payload.get('request_id', ''),
            **common_params
        )
    
    elif event_kind == 'DataResponseEvent':
        event = DataResponseEvent(
            from_agent_id=proto_event.from_agent_id,
            to_agent_id=proto_event.to_agent_id,
            request_id=payload.get('request_id', ''),
            key=payload.get('key', ''),
            data_value=payload.get('data_value'),
            success=payload.get('success', False),
            error=payload.get('error'),
            **common_params
        )
    
    elif event_kind == 'DataUpdateEvent':
        event = DataUpdateEvent(
            from_agent_id=proto_event.from_agent_id,
            to_agent_id=proto_event.to_agent_id,
            source_type=payload.get('source_type', 'UNKNOWN'),
            target_type=payload.get('target_type', 'UNKNOWN'),
            key=payload.get('key', ''),
            value=payload.get('value'),
            request_id=payload.get('request_id', ''),
            **common_params
        )
    
    elif event_kind == 'DataUpdateResponseEvent':
        event = DataUpdateResponseEvent(
            from_agent_id=proto_event.from_agent_id,
            to_agent_id=proto_event.to_agent_id,
            request_id=payload.get('request_id', ''),
            key=payload.get('key', ''),
            success=payload.get('success', False),
            error=payload.get('error'),
            **common_params
        )
    
    elif event_kind == 'EndEvent':
        event = EndEvent(
            from_agent_id=proto_event.from_agent_id,
            to_agent_id=proto_event.to_agent_id,
            **payload,
            **common_params
        )
    
    else:
        # Generic event for unknown types
        event = Event(
            from_agent_id=proto_event.from_agent_id,
            to_agent_id=proto_event.to_agent_id,
            event_kind=event_kind,
            **payload,
            **common_params
        )
    
    # Add any remaining fields from payload
    for key, value in payload.items():
        if not hasattr(event, key):
            setattr(event, key, value)
    
    return event

# Add this utility function near the top of the file, after imports
def run_async_safely(coro):
    """
    Safely run an async coroutine in a synchronous context.
    Works even in threads without an existing event loop.
    """
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If no event loop exists in this thread, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    try:
        return loop.run_until_complete(coro)
    finally:
        # Don't close the loop here, as it might be used elsewhere
        pass

# Master server implementation.
class MasterServicer(agent_pb2_grpc.AgentServiceServicer):
    """gRPC service implementation for the master node."""
    
    def __init__(self, master_node):
        self.master_node = master_node
        
    async def RegisterWorker(self, request, context):
        """Handle worker registration asynchronously."""
        try:
            worker_id = request.worker_id
            address = request.address
            port = request.port
            
            # Set a cancellation callback on the context so long-running operations can be canceled.
            # context.add_done_callback(lambda _: logger.debug(f"RegisterWorker RPC for {worker_id} completed or cancelled"))
            
            # Ensure client_metadata is available.
            client_ip = context.peer().split(':')[-2].strip('[]')
            #logger.info(f"RegisterWorker request from {client_ip} for worker {worker_id} at {address}:{port}")
            
            success, message = await self.master_node.register_worker(worker_id, address, port)
            
            # Ensure the master has updated its data structures before returning.
            await asyncio.sleep(0.05)  # Wait briefly for synchronization.
            
            response = agent_pb2.RegisterWorkerResponse(success=success, message=message)
            return response
            
        except Exception as e:
            logger.error(f"Error registering worker: {e}")
            return agent_pb2.RegisterWorkerResponse(
                success=False, 
                message=f"Registration failed: {str(e)}"
            )
    
    async def Heartbeat(self, request, context):
        """Update the worker's last heartbeat time."""
        try:
            worker_id = request.worker_id
            result = await self.master_node.update_worker_heartbeat(worker_id)
            
            # Return a special response asking unregistered workers to register again.
            if result is False:
                logger.warning(f"Heartbeat from unregistered worker {worker_id}, suggesting re-registration")
                return agent_pb2.HeartbeatResponse(acknowledged=False)
                
            return agent_pb2.HeartbeatResponse(acknowledged=True)
        except Exception as e:
            logger.error(f"Error processing heartbeat from {worker_id}: {e}")
            return agent_pb2.HeartbeatResponse(acknowledged=False)
    
    async def SendEvent(self, request, context):
        """Receive and forward an event from a worker."""
        try:
            event = proto_to_event(request)
            
            # Handle data events with special routing logic
            if event.event_kind == "DataEvent":
                # Find the target and route accordingly
                target_id = event.to_agent_id
                if target_id and event.target_type == "AGENT":
                    # For agent data requests, find the worker hosting the agent
                    worker = self.master_node.find_worker_for_agent(target_id)
                    if worker:
                        # Forward to appropriate worker node
                        await send_event_to_worker(
                            worker['address'], 
                            worker['port'], 
                            event
                        )
                    else:
                        # If agent not found, create a failure response
                        from onesim.events import DataResponseEvent
                        response_event = DataResponseEvent(
                            from_agent_id="ENV",
                            to_agent_id=event.from_agent_id,
                            request_id=event.request_id,
                            key=event.key,
                            data_value=None,
                            success=False,
                            error=f"Agent {target_id} not found in the network"
                        )
                        # Send back to originator
                        origin_worker = self.master_node.find_worker_for_agent(event.from_agent_id)
                        if origin_worker:
                            await send_event_to_worker(
                                origin_worker['address'],
                                origin_worker['port'],
                                response_event
                            )
                
                elif event.target_type == "ENV":
                    # For environment data requests, use the master's environment
                    if hasattr(self.master_node, 'sim_env') and self.master_node.sim_env:
                        try:
                            # Get data from environment
                            data_value = await self.master_node.sim_env.get_data(event.key, event.default)
                            
                            # Create response
                            from onesim.events import DataResponseEvent
                            response_event = DataResponseEvent(
                                from_agent_id="ENV",
                                to_agent_id=event.from_agent_id,
                                request_id=event.request_id,
                                key=event.key,
                                data_value=data_value,
                                success=True,
                                parent_event_id=event.event_id
                            )
                            
                            # Send response back to originator
                            origin_worker = self.master_node.find_worker_for_agent(event.from_agent_id)
                            if origin_worker:
                                await send_event_to_worker(
                                    origin_worker['address'],
                                    origin_worker['port'],
                                    response_event
                                )
                            else:
                                logger.warning(
                                    f"ENV data response not sent: no worker found for agent {event.from_agent_id} "
                                    f"(key={event.key}). Agent will timeout. Check master agent_locations."
                                )
                        except Exception as e:
                            # Create failure response
                            from onesim.events import DataResponseEvent
                            response_event = DataResponseEvent(
                                from_agent_id="ENV",
                                to_agent_id=event.from_agent_id,
                                request_id=event.request_id,
                                key=event.key,
                                data_value=None,
                                success=False,
                                error=str(e),
                                parent_event_id=event.event_id
                            )
                            
                            # Send back to originator
                            origin_worker = self.master_node.find_worker_for_agent(event.from_agent_id)
                            if origin_worker:
                                await send_event_to_worker(
                                    origin_worker['address'],
                                    origin_worker['port'],
                                    response_event
                                )
                            else:
                                logger.warning(
                                    f"ENV data error response not sent: no worker for agent {event.from_agent_id} (key={event.key})."
                                )
            
            # Process the event through the master's event system
            await self.master_node.event_bus.dispatch_event(event)
            
            return agent_pb2.EventResponse(received=True)
            
        except Exception as e:
            logger.error(f"Error handling event in master: {e}")
            return agent_pb2.EventResponse(received=False)
        
    async def SendStorageEvent(self, request, context):
        """Receive storage event data from a worker."""
        try:
            event_type = request.event_type
            source_type = request.source_type
            source_id = request.source_id
            target_type = request.target_type if request.target_type else None
            target_id = request.target_id if request.target_id else None
            payload = json.loads(request.payload_json) if request.payload_json else {}
            
            event_data = {
                'event_type': event_type,
                'source_type': source_type,
                'source_id': source_id,
                'payload': payload
            }
            
            if target_type and target_id:
                event_data['target_type'] = target_type
                event_data['target_id'] = target_id
            
            # Queue the event for storage
            if hasattr(self.master_node.sim_env, 'queue_event'):
                await self.master_node.sim_env.queue_event(event_data)
            else:
                logger.warning("Master sim_env does not have queue_event method")
                
            return agent_pb2.StorageEventResponse(received=True)
        except Exception as e:
            logger.error(f"Error processing storage event: {e}")
            return agent_pb2.StorageEventResponse(received=False)
    
    async def SendDecisionRecord(self, request, context):
        """Handle a decision record request."""
        try:
            decision_data = {
                'agent_id': request.agent_id,
                'prompt': request.prompt,
                'output': request.output,
                'processing_time': request.processing_time,
                'timestamp': datetime.now().isoformat(),
                'decision_id': str(uuid.uuid4()),
            }
            
            if request.context_json:
                try:
                    context_data = json.loads(request.context_json)
                    decision_data['context'] = context_data
                except json.JSONDecodeError:
                    logger.error(f"Invalid context JSON in decision record: {request.context_json}")
            
            # Forward the decision record to sim_env if available.
            if hasattr(self.master_node, 'sim_env') and self.master_node.sim_env:
                if hasattr(self.master_node.sim_env, 'queue_decision'):
                    await self.master_node.sim_env.queue_decision(decision_data)
                    
            return agent_pb2.DecisionRecordResponse(received=True)
        except Exception as e:
            logger.error(f"Error processing decision record: {e}")
            return agent_pb2.DecisionRecordResponse(received=False)

    async def GetEnvData(self, request, context):
        """Handle an environment data request."""
        try:
            key = request.key
            default = None
            
            if request.default_value_json:
                try:
                    default = json.loads(request.default_value_json)
                except json.JSONDecodeError:
                    logger.error(f"Invalid default value JSON: {request.default_value_json}")
            
            # Read data from sim_env if available.
            value = None
            success = False
            error = "Environment reference not available"
            
            if hasattr(self.master_node, 'sim_env') and self.master_node.sim_env:
                if hasattr(self.master_node.sim_env, 'get_data'):
                    try:
                        # Get data from environment asynchronously
                        value = await self.master_node.sim_env.get_data(key, default)
                        success = True
                        error = ""
                    except Exception as e:
                        logger.error(f"Error getting data from environment: {e}")
                        error = str(e)
                else:
                    error = "Environment does not support get_data"
            
            # Serialize value to JSON
            value_json = ""
            if value is not None:
                try:
                    value_json = json.dumps(value)
                except (TypeError, ValueError) as e:
                    logger.error(f"Error serializing environment data: {e}")
                    success = False
                    error = f"Error serializing data: {str(e)}"
            
            return agent_pb2.EnvDataResponse(
                success=success,
                value_json=value_json,
                error=error
            )
        except Exception as e:
            logger.error(f"Error in GetEnvData: {e}")
            return agent_pb2.EnvDataResponse(
                success=False,
                value_json="",
                error=str(e)
            )
    
    async def UpdateEnvData(self, request, context):
        """Handle an environment data update request."""
        try:
            key = request.key
            value = None
            
            if request.value_json:
                try:
                    value = json.loads(request.value_json)
                except json.JSONDecodeError:
                    logger.error(f"Invalid value JSON: {request.value_json}")
                    return agent_pb2.EnvDataUpdateResponse(
                        success=False,
                        error="Invalid JSON value"
                    )
            
            # Update sim_env data if available.
            success = False
            error = "Environment reference not available"
            
            if hasattr(self.master_node, 'sim_env') and self.master_node.sim_env:
                if hasattr(self.master_node.sim_env, 'update_data'):
                    try:
                        # Update data in environment asynchronously
                        success = await self.master_node.sim_env.update_data(key, value)
                        error = "" if success else "Update failed"
                    except Exception as e:
                        logger.error(f"Error updating data in environment: {e}")
                        error = str(e)
                else:
                    error = "Environment does not support update_data"
            
            return agent_pb2.EnvDataUpdateResponse(
                success=success,
                error=error
            )
        except Exception as e:
            logger.error(f"Error in UpdateEnvData: {e}")
            return agent_pb2.EnvDataUpdateResponse(
                success=False,
                error=str(e)
            )
    
    async def StopSimulation(self, request, context):
        """Handle a simulation termination request."""
        try:
            worker_id = request.worker_id
            reason = request.reason
            
            logger.info(f"Received simulation stop request from worker {worker_id}: {reason}")
            
            # Stop the simulation through sim_env if available.
            success = False
            message = "Environment reference not available"
            
            if hasattr(self.master_node, 'sim_env') and self.master_node.sim_env:
                if hasattr(self.master_node.sim_env, 'stop_simulation'):
                    try:
                        # Stop simulation asynchronously
                        await self.master_node.sim_env.stop_simulation()
                        success = True
                        message = "Simulation stop initiated"
                    except Exception as e:
                        logger.error(f"Error stopping simulation: {e}")
                        message = str(e)
                else:
                    message = "Environment does not support stop_simulation"
            
            return agent_pb2.SimulationStopResponse(
                acknowledged=success,
                message=message
            )
        except Exception as e:
            logger.error(f"Error in StopSimulation: {e}")
            return agent_pb2.SimulationStopResponse(
                acknowledged=False,
                message=str(e)
            )

    async def GetAgentData(self, request, context):
        """Handle an agent data request."""
        try:
            agent_id = request.agent_id
            key = request.key
            default = None
            
            if request.default_value_json:
                try:
                    default = json.loads(request.default_value_json)
                except json.JSONDecodeError:
                    logger.error(f"Invalid default value JSON: {request.default_value_json}")
            
            # Read data from sim_env if available.
            value = None
            success = False
            error = "Environment reference not available"
            
            if hasattr(self.master_node, 'sim_env') and self.master_node.sim_env:
                if hasattr(self.master_node.sim_env, 'get_agent_data'):
                    try:
                        # Get agent data from environment asynchronously
                        value = await self.master_node.sim_env.get_agent_data(agent_id, key, default)
                        success = True
                        error = ""
                    except Exception as e:
                        logger.error(f"Error getting agent data from environment: {e}")
                        error = str(e)
                else:
                    error = "Environment does not support get_agent_data"
            
            # Serialize value to JSON
            value_json = ""
            if value is not None:
                try:
                    value_json = json.dumps(value)
                except (TypeError, ValueError) as e:
                    logger.error(f"Error serializing agent data: {e}")
                    success = False
                    error = f"Error serializing data: {str(e)}"
            
            return agent_pb2.AgentDataResponse(
                success=success,
                value_json=value_json,
                error=error
            )
        except Exception as e:
            logger.error(f"Error in GetAgentData: {e}")
            return agent_pb2.AgentDataResponse(
                success=False,
                value_json="",
                error=str(e)
            )
    
    async def GetAgentDataByType(self, request, context):
        """Handle a request for data from agents of a specified type."""
        try:
            agent_type = request.agent_type
            key = request.key
            default = None
            
            if request.default_value_json:
                try:
                    default = json.loads(request.default_value_json)
                except json.JSONDecodeError:
                    logger.error(f"Invalid default value JSON: {request.default_value_json}")
            
            # Read data from sim_env if available.
            values = {}
            success = False
            error = "Environment reference not available"
            
            if hasattr(self.master_node, 'sim_env') and self.master_node.sim_env:
                if hasattr(self.master_node.sim_env, 'get_agent_data_by_type'):
                    try:
                        # Get agent data by type from environment asynchronously
                        values = await self.master_node.sim_env.get_agent_data_by_type(agent_type, key, default)
                        success = True
                        error = ""
                    except Exception as e:
                        logger.error(f"Error getting agent data by type from environment: {e}")
                        error = str(e)
                else:
                    error = "Environment does not support get_agent_data_by_type"
            
            # Serialize values to JSON
            values_json = "{}"
            if values:
                try:
                    values_json = json.dumps(values)
                except (TypeError, ValueError) as e:
                    logger.error(f"Error serializing agent data by type: {e}")
                    success = False
                    error = f"Error serializing data: {str(e)}"
            
            return agent_pb2.AgentDataByTypeResponse(
                success=success,
                values_json=values_json,
                error=error
            )
        except Exception as e:
            logger.error(f"Error in GetAgentDataByType: {e}")
            return agent_pb2.AgentDataByTypeResponse(
                success=False,
                values_json="{}",
                error=str(e)
            )

    async def GetTokenUsage(self, request, context):
        """Handle a request for a node's token usage."""
        try:
            worker_id = request.worker_id
            try:
                token_stats = await self.master_node.get_token_usage(worker_id)
                return agent_pb2.TokenUsageResponse(
                    success=True,
                    token_stats_json=json.dumps(token_stats),
                    error=""
                )
            except Exception as e:
                logger.error(f"Error getting token usage from master: {e}")
                return agent_pb2.TokenUsageResponse(
                    success=False,
                    token_stats_json="{}",
                    error=str(e)
                )
        except Exception as e:
            logger.error(f"Error in GetTokenUsage: {e}")
            return agent_pb2.TokenUsageResponse(
                success=False,
                token_stats_json="{}",
                error=str(e)
            )

    async def SendStorageEventBatch(self, request, context):
        """Receive a batch of storage events from a worker."""
        try:
            processed_count = 0
            
            for event_request in request.events:
                event_type = event_request.event_type
                source_type = event_request.source_type
                source_id = event_request.source_id
                target_type = event_request.target_type if event_request.target_type else None
                target_id = event_request.target_id if event_request.target_id else None
                payload = json.loads(event_request.payload_json) if event_request.payload_json else {}
                
                event_data = {
                    'event_type': event_type,
                    'source_type': source_type,
                    'source_id': source_id,
                    'payload': payload
                }
                
                if target_type and target_id:
                    event_data['target_type'] = target_type
                    event_data['target_id'] = target_id
                
                # Queue the event for storage
                if hasattr(self.master_node.sim_env, 'queue_event'):
                    await self.master_node.sim_env.queue_event(event_data)
                    processed_count += 1
                else:
                    logger.warning("Master sim_env does not have queue_event method")
                    
            return agent_pb2.StorageEventBatchResponse(
                received=True,
                processed_count=processed_count
            )
        except Exception as e:
            logger.error(f"Error processing batch storage events: {e}")
            return agent_pb2.StorageEventBatchResponse(
                received=False,
                processed_count=0,
                error=str(e)
            )
    
    async def SendDecisionRecordBatch(self, request, context):
        """Handle a batch of decision records."""
        try:
            processed_count = 0
            
            for decision_request in request.decisions:
                decision_data = {
                    'agent_id': decision_request.agent_id,
                    'prompt': decision_request.prompt,
                    'output': decision_request.output,
                    'processing_time': decision_request.processing_time,
                    'timestamp': datetime.now().isoformat(),
                    'decision_id': str(uuid.uuid4()),
                }
                
                if decision_request.context_json:
                    try:
                        context_data = json.loads(decision_request.context_json)
                        decision_data['context'] = context_data
                    except json.JSONDecodeError:
                        logger.error(f"Invalid context JSON in batch decision record: {decision_request.context_json}")
                
                # Forward the decision record to sim_env if available.
                if hasattr(self.master_node, 'sim_env') and self.master_node.sim_env:
                    if hasattr(self.master_node.sim_env, 'queue_decision'):
                        await self.master_node.sim_env.queue_decision(decision_data)
                        processed_count += 1
                        
            return agent_pb2.DecisionRecordBatchResponse(
                received=True,
                processed_count=processed_count
            )
        except Exception as e:
            logger.error(f"Error processing decision record batch: {e}")
            return agent_pb2.DecisionRecordBatchResponse(
                received=False, 
                processed_count=0,
                error=str(e)
            )

    async def CollectDataBatch(self, request, context):
        """Handles a batch data collection request from the Master."""
        try:
            agent_type = request.agent_type
            data_key = request.data_key
            default_value = None
            if request.default_value_json:
                try:
                    default_value = json.loads(request.default_value_json)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid default_value_json in CollectDataBatch: {request.default_value_json}")
            
            # Delegate to worker_node to collect data from its local agents
            collected_data = await self.master_node.collect_local_agent_data_batch(
                agent_type, data_key, default_value
            )
            
            collected_data_json = json.dumps(collected_data)
            return agent_pb2.BatchDataResponse(
                success=True,
                collected_data_json=collected_data_json
            )
        except Exception as e:
            logger.error(f"Error in MasterServicer.CollectDataBatch: {e}")
            return agent_pb2.BatchDataResponse(
                success=False,
                error_message=str(e)
            )

    async def LocateAgent(self, request, context): # Added LocateAgent to MasterServicer
        """Handle an agent location request."""
        try:
            agent_id = request.agent_id
            location = self.master_node.find_worker_for_agent(agent_id)
            if location:
                return agent_pb2.LocateAgentResponse(
                    success=True,
                    worker_address=location['address'],
                    worker_port=location['port']
                )
            else:
                return agent_pb2.LocateAgentResponse(success=False, error_message="Agent not found")
        except Exception as e:
            logger.error(f"Error in LocateAgent: {e}")
            return agent_pb2.LocateAgentResponse(success=False, error_message=str(e))

# Worker server implementation.
class WorkerServicer(agent_pb2_grpc.AgentServiceServicer):
    def __init__(self, worker_node):
        self.worker_node = worker_node
        # self.p2p_reply_info_cache = {} # Cache for P2P: request_id -> (reply_addr, reply_port)

    async def SendEvent(self, request, context):
        """Receive an event from the master or another worker."""
        try:
            event = proto_to_event(request) # proto_to_event should handle reply_to fields

            # # Logic for P2P reply handling (if needed here, or primarily in GeneralAgent)
            # if hasattr(request, 'reply_to_worker_address') and request.reply_to_worker_address:
            #     # This is an initial request from another worker, store its reply info
            #     # Assuming request_id is part of the event payload or a direct field
            #     request_id = getattr(event, 'request_id', None) # or event.request_id if always present
            #     if request_id:
            #         self.p2p_reply_info_cache[request_id] = (request.reply_to_worker_address, request.reply_to_worker_port)
            #         logger.debug(f"Cached P2P reply info for {request_id} to {request.reply_to_worker_address}:{request.reply_to_worker_port}")

            await self.worker_node.handle_event(event)
            return agent_pb2.EventResponse(received=True)
        except Exception as e:
            logger.error(f"Error processing event in WorkerServicer: {e}")
            return agent_pb2.EventResponse(received=False)

    async def StopSimulation(self, request, context):
        """Handle a simulation termination request from the master."""
        try:
            reason = request.reason

            # Respond immediately, then shut down asynchronously so closing gRPC does not prevent delivery of the response.
            asyncio.create_task(self.worker_node.handle_termination_signal(reason))
            return agent_pb2.SimulationStopResponse(
                acknowledged=True, message="Acknowledged"
            )
        except Exception as e:
            logger.error(f"Error stopping simulation: {e}")
            return agent_pb2.SimulationStopResponse(
                acknowledged=False,
                message=str(e)
            )

    async def CreateAgent(self, request, context):
        """Receive an agent creation request."""
        try:
            # Parse agent data.
            agent_data = {
                "type": request.agent_type,
                "id": request.agent_id,
                "config": json.loads(request.config_json)
            }

            # Create the agent.
            agent_id = await self.worker_node.create_agent(agent_data)

            return agent_pb2.CreateAgentResponse(
                success=True,
                agent_id=agent_id
            )
        except Exception as e:
            logger.error(f"Error creating agent: {e}")
            return agent_pb2.CreateAgentResponse(
                success=False,
                message=str(e)
            )

    async def CreateAgentsBatch(self, request, context):
        """Receive a batch agent creation request."""
        try:
            # Parse agent configurations.
            agent_configs = []
            for config_json in request.configs_json:
                agent_configs.append(json.loads(config_json))

            # Create agents in a batch.
            agent_ids = await self.worker_node.create_agents_batch(agent_configs)

            # Ensure all agent IDs are strings
            string_agent_ids = [str(agent_id) for agent_id in agent_ids]

            return agent_pb2.CreateAgentsBatchResponse(
                success=True,
                agent_ids=string_agent_ids,
                message="Agents created successfully"
            )
        except Exception as e:
            logger.error(f"Error creating agents batch: {e}")
            return agent_pb2.CreateAgentsBatchResponse(
                success=False,
                message=str(e)
            )

    async def GetTokenUsage(self, request, context):
        """Handle a request for a node's token usage."""
        try:
            worker_id = request.worker_id
            try:
                token_stats = await self.worker_node.get_token_usage(worker_id)
                return agent_pb2.TokenUsageResponse(
                    success=True,
                    token_stats_json=json.dumps(token_stats),
                    error=""
                )
            except Exception as e:
                logger.error(f"Error getting token usage from worker: {e}")
                return agent_pb2.TokenUsageResponse(
                    success=False,
                    token_stats_json="{}",
                    error=str(e)
                )
        except Exception as e:
            logger.error(f"Error in GetTokenUsage: {e}")
            return agent_pb2.TokenUsageResponse(
                success=False,
                token_stats_json="{}",
                error=str(e)
            )

    async def SendEventBatch(self, request, context): # This is for P2P batching if enabled
        # ... (existing code)
        pass # Placeholder for brevity

    async def CollectDataBatch(self, request, context):
        """Handles a batch data collection request from the Master."""
        try:
            agent_type = request.agent_type
            data_key = request.data_key
            default_value = None
            if request.default_value_json:
                try:
                    default_value = json.loads(request.default_value_json)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid default_value_json in CollectDataBatch: {request.default_value_json}")

            collected_data = await self.worker_node.collect_local_agent_data_batch(
                agent_type, data_key, default_value
            )

            collected_data_json = json.dumps(collected_data)
            return agent_pb2.BatchDataResponse(
                success=True,
                collected_data_json=collected_data_json
            )
        except Exception as e:
            logger.error(f"Error in WorkerServicer.CollectDataBatch: {e}")
            return agent_pb2.BatchDataResponse(
                success=False,
                error_message=str(e)
            )

# Server creation functions.
async def create_master_server(master_node, port):
    """Create and start the master server."""
    server = grpc.aio.server(
        maximum_concurrent_rpcs=1000,  # Allow more concurrent RPCs to prevent queue blocking.
        options=[
            ('grpc.keepalive_time_ms', 10000),             # Send a keepalive every 10 seconds.
            ('grpc.keepalive_timeout_ms', 5000),          # Wait up to 5 seconds for a keepalive response.
            ('grpc.http2.max_pings_without_data', 0),     # Allow pings without data.
            ('grpc.http2.max_ping_strikes', 0),           # Do not disconnect because of ping strikes.
            ('grpc.http2.min_ping_interval_without_data_ms', 30000),
            ('grpc.keepalive_permit_without_calls', 1),   # Allow keepalive while idle.
            ('grpc.max_connection_idle_ms', 900000),      # Connection idle timeout of 15 minutes.
            ('grpc.max_connection_age_ms', 3600000),       # Maximum connection age of 60 minutes.
            ('grpc.max_connection_age_grace_ms', 120000),  # Connection shutdown grace period of 2 minutes.
            ('grpc.server_handshake_timeout_ms', 60000),  # Extend the handshake timeout.
            ('grpc.max_concurrent_streams', 1000),        # Allow more concurrent streams.
            ('grpc.max_receive_message_length', 10 * 1024 * 1024),  # Raise the message size limit to 10 MB.
            ('grpc.initial_reconnect_backoff_ms', 100),   # Initial reconnection backoff.
            ('grpc.max_reconnect_backoff_ms', 10000)      # Maximum reconnection backoff.
        ]
    )

    agent_pb2_grpc.add_AgentServiceServicer_to_server(
        MasterServicer(master_node), server
    )
    
    server.add_insecure_port(f'[::]:{port}')
    await server.start()
    logger.info(f"Master server started on port {port}")
    return server

async def create_worker_server(worker_node, port):
    """Create and start the worker server."""
    servicer_instance = WorkerServicer(worker_node)
    server = grpc.aio.server(
        maximum_concurrent_rpcs=1000,  # Allow more concurrent RPCs to prevent queue blocking.
        options=[
            ('grpc.keepalive_time_ms', 10000),             # Send a keepalive every 10 seconds.
            ('grpc.keepalive_timeout_ms', 5000),          # Wait up to 5 seconds for a keepalive response.
            ('grpc.http2.max_pings_without_data', 0),     # Allow pings without data.
            ('grpc.http2.max_ping_strikes', 0),           # Do not disconnect because of ping strikes.
            ('grpc.http2.min_ping_interval_without_data_ms', 30000),
            ('grpc.keepalive_permit_without_calls', 1),   # Allow keepalive while idle.
            ('grpc.max_connection_idle_ms', 900000),      # Connection idle timeout of 15 minutes.
            ('grpc.max_connection_age_ms', 3600000),       # Maximum connection age of 60 minutes.
            ('grpc.max_connection_age_grace_ms', 120000),  # Connection shutdown grace period of 2 minutes.
            ('grpc.server_handshake_timeout_ms', 60000),  # Extend the handshake timeout.
            ('grpc.max_concurrent_streams', 1000),        # Allow more concurrent streams.
            ('grpc.max_receive_message_length', 10 * 1024 * 1024),  # Raise the message size limit to 10 MB.
            ('grpc.initial_reconnect_backoff_ms', 100),   # Initial reconnection backoff.
            ('grpc.max_reconnect_backoff_ms', 10000)      # Maximum reconnection backoff.
        ]
    )
    
    agent_pb2_grpc.add_AgentServiceServicer_to_server(
        servicer_instance, server
    )
    
    # Store the servicer instance on the worker_node for later access to its cache
    if hasattr(worker_node, 'set_servicer_instance'):
        worker_node.set_servicer_instance(servicer_instance)
    else:
        logger.warning("WorkerNode does not have set_servicer_instance method. P2P reply cache access might fail.")

    server.add_insecure_port(f'[::]:{port}')
    await server.start()
    logger.info(f"Worker server started on port {port}")
    return server


def get_host_ip():
    """Get a local IP address suitable for network communication."""
    import socket
    try:
        # Use a temporary socket to determine the IP used for external communication.
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        # Fallback.
        return socket.gethostbyname(socket.gethostname())

# Asynchronous client functions.
async def register_with_master(master_address, master_port, worker_id, worker_port, worker_address=None):
    """Register the worker with the master."""
    # Try registration at most five times.
    max_retries = 5
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            if worker_address is None:
                # Use 127.0.0.1 by default to connect to a local master.
                if master_address == "127.0.0.1" or master_address == "localhost":
                    worker_address = "127.0.0.1"
                # Otherwise try to obtain an external IP address.
                else:
                    worker_address = get_host_ip()
                
            
            request = agent_pb2.RegisterWorkerRequest(
                worker_id=worker_id,
                address=worker_address,  
                port=worker_port
            )
            
            # Use a longer timeout and higher priority.
            response = await connection_manager.with_stub(
                master_address, 
                master_port, 
                agent_pb2_grpc.AgentServiceStub, 
                'RegisterWorker', 
                request
            )
            
            return response.success
            
        except Exception as e:
            retry_count += 1
            logger.error(f"Error registering with master (attempt {retry_count}/{max_retries}): {e}")
            
            if retry_count < max_retries:
                # Exponential backoff.
                wait_time = 1.0 * (2 ** retry_count)
                wait_time = min(wait_time, 30.0)  # Wait at most 30 seconds.
                logger.info(f"Retrying registration in {wait_time:.1f} seconds...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Maximum registration attempts reached. Failed to register worker.")
                return False
                
    return False

async def send_heartbeat_to_master(master_address, master_port, worker_id):
    """Send a heartbeat to the master."""
    try:
        request = agent_pb2.HeartbeatRequest(
            worker_id=worker_id,
            timestamp=int(time.time() * 1000)
        )
        
        # Use a shorter timeout.
        response = await connection_manager.with_stub(
            master_address, 
            master_port, 
            agent_pb2_grpc.AgentServiceStub, 
            'Heartbeat', 
            request
        )
        
        # Check the acknowledgment; registration may need to be repeated.
        if not response.acknowledged:
            logger.warning("Heartbeat not acknowledged by master, may need to re-register")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error sending heartbeat: {e}")
        return False

async def send_event_to_worker(worker_address, worker_port, event):
    """Send an event to a worker node."""
    try:
        proto_event = event_to_proto(event)

        try:
            response = await connection_manager.with_stub(
                worker_address,
                worker_port,
                agent_pb2_grpc.AgentServiceStub,
                'SendEvent',
                proto_event,
            )
        except RuntimeError as rt:
            # Circuit breaker open or manager in shutdown; suppress and treat as delivered during shutdown
            logger.warning(
                f"SendEvent suppressed for {worker_address}:{worker_port}: {rt}"
            )
            return False

        return response.received
    except Exception as e:
        try:
            import grpc

            if (
                isinstance(e, grpc.aio.AioRpcError)
                and e.code() == grpc.StatusCode.UNAVAILABLE
            ):
                logger.warning(
                    f"Worker {worker_address}:{worker_port} unavailable for SendEvent during shutdown: {e.details()}"
                )
                return False
        except Exception:
            pass
        logger.error(f"Error sending event to worker {worker_address}:{worker_port}: {e}")
        return False

async def send_termination_to_worker(worker_address, worker_port, event):
    """Send a termination signal to a worker node."""
    try:
        logger.info(f"Sending termination signal to worker at {worker_address}:{worker_port}")

        request = agent_pb2.SimulationStopRequest(
            worker_id="MASTER",
            reason="simulation_terminated",
            timestamp=int(time.time() * 1000)
        )

        try:
            response = await connection_manager.with_stub(
                worker_address,
                worker_port,
                agent_pb2_grpc.AgentServiceStub,
                'StopSimulation',
                request,
            )
        except RuntimeError as rt:
            # Circuit breaker open or manager in shutdown state; treat as acknowledged during shutdown
            logger.warning(
                f"Termination suppressed for {worker_address}:{worker_port}: {rt}"
            )
            return True

        if response.acknowledged:
            logger.info(f"Worker at {worker_address}:{worker_port} acknowledged termination")
        else:
            logger.warning(f"Worker at {worker_address}:{worker_port} did not acknowledge termination: {response.message}")

        return response.acknowledged
    except Exception as e:
        # Downgrade expected UNAVAILABLE during shutdown to warning to reduce noise
        try:
            import grpc

            if (
                isinstance(e, grpc.aio.AioRpcError)
                and e.code() == grpc.StatusCode.UNAVAILABLE
            ):
                logger.warning(
                    f"Worker {worker_address}:{worker_port} already unavailable during shutdown: {e.details()}"
                )
                return True
        except Exception:
            pass
        logger.error(f"Error sending termination to worker {worker_address}:{worker_port}: {e}")
        return False

async def send_event_to_master(master_address, master_port, event):
    """Send an event to the master."""
    try:
        request = event_to_proto(event)
        
        response = await connection_manager.with_stub(
            master_address, 
            master_port, 
            agent_pb2_grpc.AgentServiceStub, 
            'SendEvent', 
            request
        )
        
        return response.received
    except Exception as e:
        logger.error(f"Error sending event to master: {e}")
        return False

async def create_agent_on_worker(worker_address, worker_port, agent_data):
    """Create an agent on a worker."""
    try:
        request = agent_pb2.CreateAgentRequest(
            agent_type=agent_data["type"],
            agent_id=agent_data["id"],
            config_json=json.dumps(agent_data["config"])
        )
        
        response = await connection_manager.with_stub(
            worker_address, 
            worker_port, 
            agent_pb2_grpc.AgentServiceStub, 
            'CreateAgent', 
            request
        )
        
        return response.success
    except Exception as e:
        logger.error(f"Error creating agent on worker: {e}")
        return False

async def create_agents_batch_on_worker(worker_address, worker_port, agent_configs):
    """Create agents on a worker in a batch."""
    try:
        # Serialize each configuration as JSON.
        configs_json = [json.dumps(config) for config in agent_configs]
        
        request = agent_pb2.CreateAgentsBatchRequest(
            configs_json=configs_json
        )
        
        response = await connection_manager.with_stub(
            worker_address, 
            worker_port, 
            agent_pb2_grpc.AgentServiceStub, 
            'CreateAgentsBatch', 
            request
        )
        
        return response.success
    except Exception as e:
        logger.error(f"Error creating agents batch on worker: {e}")
        return False

# Functions for sending data from workers to the master.
async def send_storage_event_to_master(master_address, master_port, event_data):
    """Send storage event data to the master."""
    # If event_data is a list, use batch sending
    if isinstance(event_data, list):
        return await send_storage_event_batch_to_master(master_address, master_port, event_data)
        
    # Original single-event implementation
    try:
        # Build the request.
        request = agent_pb2.StorageEventRequest(
            event_type=event_data.get('event_type', ''),
            source_type=event_data.get('source_type', ''),
            source_id=event_data.get('source_id', ''),
            target_type=event_data.get('target_type', ''),
            target_id=event_data.get('target_id', ''),
            payload_json=json.dumps(event_data.get('payload', {}))
        )
        
        response = await connection_manager.with_stub(
            master_address, 
            master_port, 
            agent_pb2_grpc.AgentServiceStub, 
            'SendStorageEvent', 
            request
        )
        
        return response.received
    except Exception as e:
        logger.error(f"Error sending storage event to master: {e}")
        return False

async def send_decision_record_to_master(master_address, master_port, decision_data):
    """Send decision record data to the master."""
    # If decision_data is a list, use batch sending
    if isinstance(decision_data, list):
        return await send_decision_record_batch_to_master(master_address, master_port, decision_data)
        
    # Original single-decision implementation
    try:
        # Build the request.
        request = agent_pb2.DecisionRecordRequest(
            agent_id=decision_data.get('agent_id', ''),
            prompt=decision_data.get('prompt', ''),
            output=decision_data.get('output', ''),
            processing_time=decision_data.get('processing_time', 0.0),
            context_json=json.dumps(decision_data.get('context', {}))
        )
        
        response = await connection_manager.with_stub(
            master_address, 
            master_port, 
            agent_pb2_grpc.AgentServiceStub, 
            'SendDecisionRecord', 
            request
        )
        
        return response.received
    except Exception as e:
        logger.error(f"Error sending decision record to master: {e}")
        return False

async def send_storage_event_batch_to_master(master_address, master_port, event_data_list):
    """Send storage event data to the master in a batch."""
    try:
        if not event_data_list:
            return True  # Nothing to send
            
        # Build the batch request.
        event_requests = []
        for event_data in event_data_list:
            request = agent_pb2.StorageEventRequest(
                event_type=event_data.get('event_type', ''),
                source_type=event_data.get('source_type', ''),
                source_id=event_data.get('source_id', ''),
                target_type=event_data.get('target_type', ''),
                target_id=event_data.get('target_id', ''),
                payload_json=json.dumps(event_data.get('payload', {}))
            )
            event_requests.append(request)
        
        batch_request = agent_pb2.StorageEventBatchRequest(events=event_requests)
        
        response = await connection_manager.with_stub(
            master_address, 
            master_port, 
            agent_pb2_grpc.AgentServiceStub, 
            'SendStorageEventBatch', 
            batch_request
        )
        
        if not response.received:
            logger.warning(f"Failed to send batch of {len(event_data_list)} events: {response.error}")
            
        return response.received
    except Exception as e:
        logger.error(f"Error sending storage event batch to master: {e}")
        return False

async def send_decision_record_batch_to_master(master_address, master_port, decision_data_list):
    """Send decision record data to the master in a batch."""
    try:
        if not decision_data_list:
            return True  # Nothing to send
            
        # Build the batch request.
        decision_requests = []
        for decision_data in decision_data_list:
            request = agent_pb2.DecisionRecordRequest(
                agent_id=decision_data.get('agent_id', ''),
                prompt=decision_data.get('prompt', ''),
                output=decision_data.get('output', ''),
                processing_time=decision_data.get('processing_time', 0.0),
                context_json=json.dumps(decision_data.get('context', {}))
            )
            decision_requests.append(request)
        
        batch_request = agent_pb2.DecisionRecordBatchRequest(decisions=decision_requests)
        
        response = await connection_manager.with_stub(
            master_address, 
            master_port, 
            agent_pb2_grpc.AgentServiceStub, 
            'SendDecisionRecordBatch', 
            batch_request
        )
        
        if not response.received:
            logger.warning(f"Failed to send batch of {len(decision_data_list)} decisions: {response.error}")
            
        return response.received
    except Exception as e:
        logger.error(f"Error sending decision record batch to master: {e}")
        return False

async def send_data_request(target_address, target_port, data_event):
    """
    Send data request to another node.
    
    Args:
        target_address: Target node address
        target_port: Target node port
        data_event: DataEvent object containing the request
    
    Returns:
        bool: Whether the request was sent successfully
    """
    try:
        # Convert DataEvent to EventRequest proto
        event_request = event_to_proto(data_event)
        
        # Send request using connection manager
        response = await connection_manager.with_stub(
            target_address,
            target_port,
            agent_pb2_grpc.AgentServiceStub,
            'SendEvent',
            event_request
        )
            
        if response.received:
            # The actual data will be returned via a DataResponseEvent
            # which will be processed by the event handling system
            return True
        else:
            logger.warning(f"Failed to send data request to {target_address}:{target_port}")
            return False
                
    except Exception as e:
        logger.error(f"Error sending data request: {e}")
        return False

async def send_data_update(target_address, target_port, data_update_event):
    """
    Send a data update request to a remote node
    
    Args:
        target_address: Address of the target node
        target_port: Port of the target node
        data_update_event: The DataUpdateEvent to send
        
    Returns:
        The response event or None on failure
    """
    try:
        # Convert event to proto
        proto_event = event_to_proto(data_update_event)
        
        # Send event using connection manager
        response = await connection_manager.with_stub(
            target_address,
            target_port,
            agent_pb2_grpc.AgentServiceStub,
            'SendEvent',
            proto_event
        )
        
        if response and response.received:
            return True
        else:
            logger.warning(f"Failed to send data update request to {target_address}:{target_port}")
            return None
    except Exception as e:
        logger.error(f"Error sending data update request: {e}")
        return None

async def acquire_lock(master_address: str, master_port: int, lock_id: str, node_id: str, timeout: float) -> bool:
    """
    Acquire a distributed lock through the master node.
    
    Args:
        master_address: Master node address
        master_port: Master node port
        lock_id: Unique identifier for the lock
        node_id: ID of the node requesting the lock
        timeout: Lock timeout in seconds
        
    Returns:
        bool: True if lock was acquired, False otherwise
    """
    try:
        async with grpc.aio.insecure_channel(f"{master_address}:{master_port}") as channel:
            stub = agent_pb2_grpc.AgentServiceStub(channel)
            request = agent_pb2.AcquireLockRequest(
                lock_id=lock_id,
                node_id=node_id,
                timeout=timeout
            )
            response = await stub.AcquireLock(request)
            return response.success
    except Exception as e:
        logger.error(f"Error in acquire_lock: {e}")
        return False

async def release_lock(master_address: str, master_port: int, lock_id: str, node_id: str) -> bool:
    """
    Release a distributed lock through the master node.
    
    Args:
        master_address: Master node address
        master_port: Master node port
        lock_id: Unique identifier for the lock
        node_id: ID of the node releasing the lock
        
    Returns:
        bool: True if lock was released, False otherwise
    """
    try:
        async with grpc.aio.insecure_channel(f"{master_address}:{master_port}") as channel:
            stub = agent_pb2_grpc.AgentServiceStub(channel)
            request = agent_pb2.ReleaseLockRequest(
                lock_id=lock_id,
                node_id=node_id
            )
            response = await stub.ReleaseLock(request)
            return response.success
    except Exception as e:
        logger.error(f"Error in release_lock: {e}")
        return False

async def get_env_data_from_master(master_address: str, master_port: int, key: str, default: Any = None) -> Any:
    """
    Retrieve environment data from the master node.
    
    Args:
        master_address: Master node address
        master_port: Master node port
        key: Data key to retrieve
        default: Default value if data not found
        
    Returns:
        The requested data or default value
    """
    try:
        # Create data request message
        request = agent_pb2.EnvDataRequest(
            key=key,
            default_value_json=json.dumps(default) if default is not None else ""
        )
        
        # Send request using connection manager
        response = await connection_manager.with_stub(
            master_address,
            master_port,
            agent_pb2_grpc.AgentServiceStub,
            'GetEnvData',
            request
        )
        
        if response.success:
            try:
                # Deserialize the data value JSON
                if response.value_json:
                    return json.loads(response.value_json)
                else:
                    return None
            except json.JSONDecodeError:
                logger.error(f"Error deserializing data value JSON: {response.value_json}")
                return default
        else:
            logger.warning(f"Error retrieving environment data: {response.error}")
            return default
                
    except Exception as e:
        logger.error(f"Error in get_env_data_from_master: {e}")
        return default

async def update_env_data_on_master(master_address: str, master_port: int, key: str, value: Any) -> bool:
    """
    Update environment data on the master node.
    
    Args:
        master_address: Master node address
        master_port: Master node port
        key: Data key to update
        value: New value to set
        
    Returns:
        True if update was successful, False otherwise
    """
    try:
        # Create data update request message
        request = agent_pb2.EnvDataUpdateRequest(
            key=key,
            value_json=json.dumps(value)
        )
        
        # Send request using connection manager
        response = await connection_manager.with_stub(
            master_address,
            master_port,
            agent_pb2_grpc.AgentServiceStub,
            'UpdateEnvData',
            request
        )
        
        return response.success
                
    except Exception as e:
        logger.error(f"Error in update_env_data_on_master: {e}")
        return False

async def request_simulation_stop(master_address: str, master_port: int, worker_id: str) -> bool:
    """
    Request the master node to stop the simulation.
    
    Args:
        master_address: Master node address
        master_port: Master node port
        worker_id: ID of the worker node making the request
        
    Returns:
        True if request was acknowledged, False otherwise
    """
    try:
        # Create stop request message
        request = agent_pb2.SimulationStopRequest(
            worker_id=worker_id,
            reason="Worker requested stop",
            timestamp=int(time.time() * 1000)
        )
        
        # Send request using connection manager
        response = await connection_manager.with_stub(
            master_address,
            master_port,
            agent_pb2_grpc.AgentServiceStub,
            'StopSimulation',
            request
        )
        
        return response.acknowledged
                
    except Exception as e:
        logger.error(f"Error in request_simulation_stop: {e}")
        return False

async def get_agent_data_from_master(master_address: str, master_port: int, agent_id: str, key: str, default: Any = None) -> Any:
    """
    Retrieve agent data from the master node.
    
    Args:
        master_address: Master node address
        master_port: Master node port
        agent_id: ID of the agent to query
        key: Data key to retrieve
        default: Default value if data not found
        
    Returns:
        The requested data or default value
    """
    try:
        # Create data request message
        request = agent_pb2.AgentDataRequest(
            agent_id=agent_id,
            key=key,
            default_value_json=json.dumps(default) if default is not None else ""
        )
        
        # Send request using connection manager
        response = await connection_manager.with_stub(
            master_address,
            master_port,
            agent_pb2_grpc.AgentServiceStub,
            'GetAgentData',
            request
        )
        
        if response.success:
            try:
                # Deserialize the data value JSON
                if response.value_json:
                    return json.loads(response.value_json)
                else:
                    return None
            except json.JSONDecodeError:
                logger.error(f"Error deserializing agent data value JSON: {response.value_json}")
                return default
        else:
            logger.warning(f"Error retrieving agent data: {response.error}")
            return default
                
    except Exception as e:
        logger.error(f"Error in get_agent_data_from_master: {e}")
        return default

async def get_agent_data_by_type_from_master(master_address: str, master_port: int, agent_type: str, key: str, default: Any = None) -> Dict[str, Any]:
    """
    Retrieve data from agents of a specific type from the master node.
    
    Args:
        master_address: Master node address
        master_port: Master node port
        agent_type: Type of agents to query
        key: Data key to retrieve
        default: Default value if data not found
        
    Returns:
        Dictionary mapping agent IDs to their data values
    """
    try:
        # Create data request message
        request = agent_pb2.AgentDataByTypeRequest(
            agent_type=agent_type,
            key=key,
            default_value_json=json.dumps(default) if default is not None else ""
        )
        
        # Send request using connection manager
        response = await connection_manager.with_stub(
            master_address,
            master_port,
            agent_pb2_grpc.AgentServiceStub,
            'GetAgentDataByType',
            request
        )
        
        if response.success:
            try:
                # Deserialize the values JSON
                if response.values_json:
                    return json.loads(response.values_json)
                else:
                    return {}
            except json.JSONDecodeError:
                logger.error(f"Error deserializing agent data by type JSON: {response.values_json}")
                return {}
        else:
            logger.warning(f"Error retrieving agent data by type: {response.error}")
            return {}
                
    except Exception as e:
        logger.error(f"Error in get_agent_data_by_type_from_master: {e}")
        return {}

async def get_token_usage_from_worker(worker_address: str, worker_port: int, worker_id: str) -> Optional[Dict[str, Any]]:
    """
    Get token usage statistics from a worker node.
    
    Args:
        worker_address: Worker node address
        worker_port: Worker node port
        worker_id: Worker node ID
        
    Returns:
        Optional[Dict[str, Any]]: Token usage statistics or None if request failed
    """
    try:
        # Create request message
        request = agent_pb2.TokenUsageRequest(
            worker_id=worker_id
        )
        
        # Send request using connection manager
        response = await connection_manager.with_stub(
            worker_address,
            worker_port,
            agent_pb2_grpc.AgentServiceStub,
            'GetTokenUsage',
            request
        )
        
        if response.success:
            try:
                # Deserialize token statistics JSON
                if response.token_stats_json:
                    return json.loads(response.token_stats_json)
                else:
                    return None
            except json.JSONDecodeError:
                logger.error(f"Error deserializing token stats JSON: {response.token_stats_json}")
                return None
        else:
            logger.warning(f"Error retrieving token usage from worker: {response.error}")
            return None
                
    except Exception as e:
        logger.error(f"Error in get_token_usage_from_worker: {e}")
        return None

async def collect_token_usage_from_workers(master_node) -> Dict[str, Any]:
    """
    Collect token usage statistics from all worker nodes.
    Uses concurrent requests for better performance.
    
    Args:
        master_node: Master node instance containing worker information
        
    Returns:
        Dict[str, Any]: Merged token usage statistics
    """
    # Initialize merged statistics
    merged_stats = {
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "request_count": 0,
        "model_usage": {},
        "worker_stats": {}
    }
    
    # First add the master node's token usage if available
    try:
        master_stats = await master_node.get_token_usage()
        if master_stats:
            # Record master's statistics
            merged_stats["worker_stats"]["master"] = master_stats
            
            # Add to total counts
            merged_stats["total_prompt_tokens"] += master_stats.get("total_prompt_tokens", 0)
            merged_stats["total_completion_tokens"] += master_stats.get("total_completion_tokens", 0)
            merged_stats["total_tokens"] += master_stats.get("total_tokens", 0)
            merged_stats["request_count"] += master_stats.get("request_count", 0)
            
            # Merge model usage
            for model, usage in master_stats.get("model_usage", {}).items():
                if model not in merged_stats["model_usage"]:
                    merged_stats["model_usage"][model] = {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "request_count": 0
                    }
                
                merged_stats["model_usage"][model]["prompt_tokens"] += usage.get("prompt_tokens", 0)
                merged_stats["model_usage"][model]["completion_tokens"] += usage.get("completion_tokens", 0)
                merged_stats["model_usage"][model]["total_tokens"] += usage.get("total_tokens", 0)
                merged_stats["model_usage"][model]["request_count"] += usage.get("request_count", 0)
    except Exception as e:
        logger.error(f"Error collecting token usage from master node: {e}")
    
    # Concurrently get statistics from all workers
    async def get_worker_stats(worker_id, worker_info):
        try:
            return worker_id, await get_token_usage_from_worker(
                worker_info.address,
                worker_info.port,
                worker_id
            )
        except Exception as e:
            logger.error(f"Error collecting token usage from worker {worker_id}: {e}")
            return worker_id, None
    
    # Create tasks for all workers
    tasks = []
    for worker_id, worker_info in master_node.workers.items():
        tasks.append(get_worker_stats(worker_id, worker_info))
    
    # Execute all tasks concurrently and wait for results
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Error in worker stats collection: {result}")
            continue
            
        if result is None:
            continue
            
        worker_id, worker_stats = result
        
        if worker_stats:
            # Record each worker's statistics
            merged_stats["worker_stats"][worker_id] = worker_stats
            
            # Add to total counts
            merged_stats["total_prompt_tokens"] += worker_stats.get("total_prompt_tokens", 0)
            merged_stats["total_completion_tokens"] += worker_stats.get("total_completion_tokens", 0)
            merged_stats["total_tokens"] += worker_stats.get("total_tokens", 0)
            merged_stats["request_count"] += worker_stats.get("request_count", 0)
            
            # Merge model usage
            for model, usage in worker_stats.get("model_usage", {}).items():
                if model not in merged_stats["model_usage"]:
                    merged_stats["model_usage"][model] = {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "request_count": 0
                    }
                
                merged_stats["model_usage"][model]["prompt_tokens"] += usage.get("prompt_tokens", 0)
                merged_stats["model_usage"][model]["completion_tokens"] += usage.get("completion_tokens", 0)
                merged_stats["model_usage"][model]["total_tokens"] += usage.get("total_tokens", 0)
                merged_stats["model_usage"][model]["request_count"] += usage.get("request_count", 0)
    
    return merged_stats

async def locate_agent_on_master(master_address: str, master_port: int, agent_id: str) -> Optional[Tuple[str, int]]:
    """
    Locate an agent by querying the master node.

    Args:
        master_address: Master node address
        master_port: Master node port
        agent_id: ID of the agent to locate

    Returns:
        Optional[Tuple[str, int]]: (worker_address, worker_port) if found, else None
    """
    try:
        request = agent_pb2.LocateAgentRequest(agent_id=agent_id)
        # Ensure connection_manager and with_stub are available and correctly used.
        response = await connection_manager.with_stub(
            master_address,
            master_port,
            agent_pb2_grpc.AgentServiceStub, # Make sure this stub is correctly generated and imported
            'LocateAgent',
            request
        )
        if response.success:
            return response.worker_address, response.worker_port
        else:
            logger.warning(f"Failed to locate agent {agent_id} via master: {response.error_message}")
            return None
    except Exception as e:
        logger.error(f"Error locating agent {agent_id} via master: {e}")
        return None

async def collect_data_batch_from_worker(worker_address: str, worker_port: int, agent_type: str, data_key: str, default_value: Any = None) -> Optional[Dict[str, Any]]:
    """Client function for Master to call CollectDataBatch on a Worker."""
    try:
        default_value_json = json.dumps(default_value) if default_value is not None else ""
        request_proto = agent_pb2.BatchDataRequest(
            agent_type=agent_type,
            data_key=data_key,
            default_value_json=default_value_json
        )
        
        response = await connection_manager.with_stub(
            worker_address,
            worker_port,
            agent_pb2_grpc.AgentServiceStub,
            'CollectDataBatch',
            request_proto
        )
        if response.success and response.collected_data_json:
            return json.loads(response.collected_data_json)
        elif not response.success:
            logger.error(f"CollectDataBatch failed on worker {worker_address}:{worker_port}: {response.error_message}")
            return None
        return {} # Success but no data
    except Exception as e:
        logger.error(f"Error calling CollectDataBatch on worker {worker_address}:{worker_port}: {e}")
        return None

# Utility function to gracefully shut down gRPC services
async def graceful_shutdown(servers):
    """
    Gracefully shutdown all gRPC servers and cleanup resources
    
    Args:
        servers: List of gRPC servers to shutdown
    """
    # Stop the batch processor
    batch_processor.stop()
    
    # Stop the connection manager
    await connection_manager.stop()
    
    # Gracefully stop all servers
    for server in servers:
        try:
            await server.stop(grace=5.0)
            logger.info(f"Server shutdown completed")
        except Exception as e:
            logger.error(f"Error during server shutdown: {e}")
    
    # Wait for remaining connections to close
    logger.info("All gRPC connections closed")
