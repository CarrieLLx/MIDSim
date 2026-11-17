import asyncio
from typing import Dict, List, Any, Optional, Set
from loguru import logger
import json

from onesim.distribution.node import get_node, NodeRole
from onesim.distribution.master import MasterNode

class AgentAllocator:
    """Assign agents to worker nodes in a distributed environment."""
    
    def __init__(self, master_node: MasterNode):
        self.master_node = master_node
        self.allocation_cache = {}  # Cache the assigned agent_id -> worker_id mapping.
        

    async def allocate_agents_batch(self, agent_configs: List[Dict[str, Any]]) -> Dict[str, str]:
        """Assign a batch of agents to workers, keeping interacting agents of different types together where possible.
        
        Args:
            agent_configs: List of agent configurations, each containing id and type.
            
        Returns:
            Dict[str, str]: Mapping from agent_id to worker_id.
        """
        if not self.master_node.workers:
            logger.error("No workers available for agent allocation")
            return {}
        
        worker_count = len(self.master_node.workers)
        worker_ids = list(self.master_node.workers.keys())
        if not worker_ids:
            return {}
        
        # Map agent IDs to configurations.
        id_to_config = {str(config.get("id")): config for config in agent_configs}
        
        # Collect relationships and build the interaction graph.
        relationship_graph = {}  # agent_id -> list of (target_id, weight)
        
        for config in agent_configs:
            agent_id = str(config.get("id"))
            agent_type = config.get("type", "unknown")
            if not agent_id:
                continue
                
            # Initialize graph nodes.
            if agent_id not in relationship_graph:
                relationship_graph[agent_id] = []
                
            # Add relationship edges.
            relationships = config.get("relationships", [])
            for rel in relationships:
                target_id = rel.get("target_id")
                if not target_id or target_id not in id_to_config:
                    continue
                
                # Check the target agent's type.
                target_type = id_to_config[target_id].get("type", "unknown")
                
                # Give relationships between different types greater weight to favor colocating them.
                weight = 3 if agent_type != target_type else 1
                    
                # Add a weighted edge.
                relationship_graph[agent_id].append((target_id, weight))
                
                # Ensure the target node exists in the graph.
                if target_id not in relationship_graph:
                    relationship_graph[target_id] = []
                relationship_graph[target_id].append((agent_id, weight))
        
        # Find subgraphs representing interaction groups.
        def find_communities():
            """Find closely interacting agent groups with a simplified community detection algorithm."""
            visited = set()
            communities = []
            
            # Process nodes in descending order of connection count.
            sorted_nodes = sorted(
                relationship_graph.keys(), 
                key=lambda x: sum(w for _, w in relationship_graph.get(x, []))
            )
            
            for node in sorted_nodes:
                if node in visited:
                    continue
                    
                # Find interaction groups using weighted BFS.
                community = []
                queue = [(node, 0)]  # (node_id, distance from start)
                community_visited = set()
                
                while queue:
                    current, dist = queue.pop(0)
                    if current in community_visited:
                        continue
                        
                    community_visited.add(current)
                    visited.add(current)
                    community.append(current)
                    
                    # Visit neighbors by weight, prioritizing interactions between different types.
                    neighbors = sorted(
                        relationship_graph.get(current, []),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    
                    for neighbor, weight in neighbors:
                        if neighbor not in community_visited:
                            # Reduce priority as distance increases.
                            queue.append((neighbor, dist + 1))
                
                if community:
                    communities.append(community)
            
            # Add isolated agents without relationships.
            for agent_id in id_to_config:
                if agent_id not in visited:
                    communities.append([agent_id])
                    visited.add(agent_id)
            
            return communities
        
        communities = find_communities()
        
        # Compute the ideal load and maximum allowed load per worker.
        agent_count = len(agent_configs)
        avg_load = max(1, agent_count // worker_count)
        max_load = avg_load + min(avg_load, 5)  # Maximum allowed load.
        
        # Handle oversized groups.
        final_communities = []
        for community in communities:
            if len(community) > max_load:
                # Split large groups greedily while preserving interactions between different types where possible.
                subgroups = []
                current_group = []
                agent_types_in_group = set()
                
                # Sort agents in the community by type diversity.
                def type_diversity(agent_id):
                    agent_type = id_to_config[agent_id].get("type", "unknown")
                    connected_types = set()
                    for neighbor, _ in relationship_graph.get(agent_id, []):
                        neighbor_type = id_to_config[neighbor].get("type", "unknown")
                        connected_types.add(neighbor_type)
                    return len(connected_types)
                
                sorted_community = sorted(community, key=type_diversity, reverse=True)
                
                for agent_id in sorted_community:
                    agent_type = id_to_config[agent_id].get("type", "unknown")
                    
                    # Start a new group at maximum load or when adding an agent would overrepresent one type.
                    if len(current_group) >= max_load:
                        if current_group:
                            subgroups.append(current_group)
                        current_group = [agent_id]
                        agent_types_in_group = {agent_type}
                    else:
                        current_group.append(agent_id)
                        agent_types_in_group.add(agent_type)
                
                if current_group:  # Add the last subgroup.
                    subgroups.append(current_group)
                
                # Append all subgroups to the final list.
                final_communities.extend(subgroups)
            else:
                final_communities.append(community)
        
        # Sort communities by size and type diversity.
        def community_priority(community):
            # Count distinct types in the community.
            types = set(id_to_config[aid].get("type", "unknown") for aid in community)
            # Use (type count, community size) as the priority.
            return (len(types), len(community))
        
        # Allocate large communities with high type diversity first.
        final_communities.sort(key=community_priority, reverse=True)
        
        # Initialize worker load tracking.
        worker_loads = {wid: 0 for wid in worker_ids}
        worker_type_counts = {wid: {} for wid in worker_ids}  # Track the count of each type on each worker.
        allocations = {}  # agent_id -> worker_id
        
        # Assign communities to workers.
        for community in final_communities:
            # Find the best worker.
            # 1. Consider load first.
            # 2. Prefer workers with complementary agent types.
            
            best_worker = None
            best_score = float('-inf')
            
            for worker_id in worker_ids:
                # Skip workers already at maximum load.
                if worker_loads[worker_id] + len(community) > max_load * 1.2:
                    continue
                    
                # Compute compatibility with the worker's existing types.
                compatibility_score = 0
                community_types = {}
                
                # Count each type in the community.
                for agent_id in community:
                    agent_type = id_to_config[agent_id].get("type", "unknown")
                    community_types[agent_type] = community_types.get(agent_type, 0) + 1
                
                # Check compatibility with existing types on the worker.
                for comm_type, comm_count in community_types.items():
                    # Reduce the score when this type is already present to avoid overconcentration.
                    worker_type_count = worker_type_counts[worker_id].get(comm_type, 0)
                    if worker_type_count > 0:
                        compatibility_score -= comm_count * worker_type_count
                    else:
                        # Reward diversity when this type is not yet present.
                        compatibility_score += comm_count
                
                # Load score: lower load is better.
                load_score = -worker_loads[worker_id]
                
                # Total score.
                total_score = load_score * 2 + compatibility_score
                
                if total_score > best_score:
                    best_score = total_score
                    best_worker = worker_id
            
            # If no suitable worker was found, choose the least loaded worker.
            if best_worker is None:
                best_worker = min(worker_loads.items(), key=lambda x: x[1])[0]
            
            # Assign the community to the selected worker.
            for agent_id in community:
                agent_type = id_to_config[agent_id].get("type", "unknown")
                allocations[agent_id] = best_worker
                worker_loads[best_worker] += 1
                
                # Update the worker's type counts.
                if agent_type not in worker_type_counts[best_worker]:
                    worker_type_counts[best_worker][agent_type] = 0
                worker_type_counts[best_worker][agent_type] += 1
        
        # Update the master node's location mapping and worker information.
        for agent_id, worker_id in allocations.items():
            self.master_node.agent_locations[agent_id] = worker_id
            
            worker = self.master_node.workers.get(worker_id)
            if worker:
                if not hasattr(worker, 'agent_count'):
                    worker.agent_count = 0
                if not hasattr(worker, 'agent_ids'):
                    worker.agent_ids = []
                
                if agent_id not in worker.agent_ids:
                    worker.agent_count += 1
                    worker.agent_ids.append(agent_id)
        
        # Check the load balance across workers.
        min_load = min(worker_loads.values()) if worker_loads else 0
        max_load = max(worker_loads.values()) if worker_loads else 0
        load_variance = max_load - min_load
        
        logger.info(f"Agent allocation complete: {len(allocations)} agents, {len(worker_ids)} workers")
        logger.info(f"Load: minimum {min_load}, maximum {max_load}, difference {load_variance}")
        
        self.allocation_cache.update(allocations)
        return allocations
    
    async def create_agents_on_workers(self, allocations: Dict[str, str], agent_configs: List[Dict[str, Any]]) -> bool:
        """Create the assigned agents on workers in batches.
        
        Args:
            allocations: Mapping from agent_id to worker_id.
            agent_configs: List of agent configurations.
            
        Returns:
            bool: Whether all agents were created successfully.
        """
        # Group by worker to reduce communication overhead.
        configs_by_worker = {}
        for config in agent_configs:
            agent_id = config.get("id")
            if not agent_id or agent_id not in allocations:
                continue
                
            worker_id = allocations[agent_id]
            if worker_id not in configs_by_worker:
                configs_by_worker[worker_id] = []
            configs_by_worker[worker_id].append(config)
        
        # Send creation requests to workers concurrently.
        tasks = []
        for worker_id, configs in configs_by_worker.items():
            worker = self.master_node.workers.get(worker_id)
            if not worker:
                logger.error(f"Worker {worker_id} not found")
                continue
            
            task = asyncio.create_task(
                self._create_agents_batch(
                    worker.address,
                    worker.port,
                    configs
                )
            )
            tasks.append(task)
        
        if not tasks:
            return False
        # Wait for all tasks to complete.
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return all(r is True for r in results if not isinstance(r, Exception))
    
    async def _create_agents_batch(self, worker_address: str, worker_port: int, agent_configs: List[Dict[str, Any]]) -> bool:
        """Create a batch of agents on the specified worker."""
        if not self.master_node._grpc_module:
            logger.error("gRPC module not initialized")
            return False
        agent_ids = [config.get("id") for config in agent_configs]
        try:
            return await self.master_node._grpc_module.create_agents_batch_on_worker(
                worker_address,
                worker_port,
                agent_configs
            )
        except Exception as e:
            logger.error(f"Error creating agents on worker {worker_address}:{worker_port}: {e}")
            return False 