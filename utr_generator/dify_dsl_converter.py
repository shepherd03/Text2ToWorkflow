import yaml
import json
from typing import Any, Dict, List, Optional
from .schema import UTR, Action, Resource, ControlIntent, Variable, ResourceType, ControlIntentType, VariableType

class DifyDSLConverter:
    def __init__(self, dsl_path: str):
        self.dsl_path = dsl_path
        self.raw_data = self._load_dsl()
        self.nodes = {}
        self.edges = []
        self.utr = UTR()

    def _load_dsl(self) -> Dict[str, Any]:
        with open(self.dsl_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def extract_description(self) -> str:
        """Extract workflow description as the natural language task input."""
        # Try to find description in app metadata first
        desc = self.raw_data.get('app', {}).get('description', '')
        if not desc:
            desc = self.raw_data.get('description', '')
        
        # Extract titles from nodes (if nodes are populated)
        titles = []
        if self.nodes:
            for node in self.nodes.values():
                title = node.get('data', {}).get('title')
                if title and title not in ['Start', 'End', '开始', '结束']:
                    titles.append(title)
        
        # Append titles to description to provide more context for generation
        if titles:
            steps_str = ", ".join(titles)
            if not desc or desc.strip() == "":
                app_name = self.raw_data.get('app', {}).get('name', '')
                if app_name:
                    desc = f"Create a workflow for '{app_name}' that involves: {steps_str}"
                else:
                    desc = f"Create a workflow that involves: {steps_str}"
            else:
                # If description exists, append steps as a hint
                desc = f"{desc}\nKey steps involved: {steps_str}"
            
        return desc

    def convert(self) -> UTR:
        # Check both locations for workflow definition
        workflow = self.raw_data.get('workflow', {})
        if not workflow:
            # Fallback for nested structure if any
            workflow = self.raw_data.get('app', {}).get('workflow', {})
            
        graph = workflow.get('graph', {})
        
        # If still no graph, try direct graph access
        if not graph:
            graph = self.raw_data.get('graph', {})
            
        nodes_list = graph.get('nodes', [])
        edges_list = graph.get('edges', [])
        
        # Build node map for easy access
        self.nodes = {node['id']: node for node in nodes_list}
        self.edges = edges_list

        # 1. Parse Start Node for Variables
        self._parse_variables()

        # 2. Parse Actions from Tool/LLM nodes
        self._parse_actions()

        # 3. Parse Edges for Control Intents
        self._parse_control_intents()
        
        # 4. Parse Resources (implied from tools)
        self._parse_resources()

        return self.utr

    def _parse_variables(self):
        for node_id, node in self.nodes.items():
            if node.get('data', {}).get('type') == 'start':
                variables = node.get('data', {}).get('variables', [])
                for var in variables:
                    utr_var = Variable(
                        name=var.get('variable', 'unknown'),
                        type=VariableType.string, # Default to string, map others if needed
                        source="user_input",
                        value=var.get('default', None)
                    )
                    self.utr.variables.append(utr_var)

    def _topological_sort(self) -> List[str]:
        # Simple BFS based topological sort simulation for graph traversal
        # Build adjacency list
        adj = {node_id: [] for node_id in self.nodes}
        in_degree = {node_id: 0 for node_id in self.nodes}
        
        for edge in self.edges:
            src = edge.get('source')
            tgt = edge.get('target')
            if src in adj and tgt in in_degree:
                adj[src].append(tgt)
                in_degree[tgt] += 1
                
        queue = [n for n, d in in_degree.items() if d == 0]
        sorted_nodes = []
        
        while queue:
            u = queue.pop(0)
            sorted_nodes.append(u)
            
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
                    
        # If cycles exist or disconnected components not reachable from 0-degree nodes
        # Add remaining nodes
        remaining = [n for n in self.nodes if n not in sorted_nodes]
        sorted_nodes.extend(remaining)
        
        return sorted_nodes

    def _parse_actions(self):
        # We need to determine order based on edges, but for now let's just parse all action nodes
        # and assign order later based on topological sort or simple edge following
        
        # Simple topological sort to get order
        sorted_node_ids = self._topological_sort()
        
        order_counter = 1
        for node_id in sorted_node_ids:
            node = self.nodes.get(node_id)
            if not node: continue
            
            # Use 'data.type' if available, otherwise fallback to 'type'
            node_data = node.get('data', {})
            node_type = node_data.get('type') or node.get('type')
            
            # Map custom type to more specific types if possible
            if node_type == 'custom':
                  if 'model' in node_data:
                      node_type = 'llm'
                  elif 'tool_name' in node_data:
                      node_type = 'tool'
                  elif 'code' in node_data: # code node usually has 'code' field
                      node_type = 'code'
                  elif 'url' in node_data and 'method' in node_data:
                      node_type = 'http-request'
            
            if node_type == 'tool':
                data = node.get('data', {})
                tool_name = data.get('tool_name', data.get('title', 'unknown_tool'))
                # Normalize common tool names
                tool_name_lower = tool_name.lower()
                if 'search' in tool_name_lower or 'ddgo' in tool_name_lower and 'translate' not in tool_name_lower:
                    tool_name = 'web_search'
                elif 'image' in tool_name_lower or 'diffusion' in tool_name_lower or 'dall' in tool_name_lower or 'img' in tool_name_lower:
                    tool_name = 'image_generation'
                elif 'translate' in tool_name_lower:
                    tool_name = 'translation'
                elif 'email' in tool_name_lower:
                    tool_name = 'email_service'
                elif 'crawl' in tool_name_lower or 'scrape' in tool_name_lower:
                    tool_name = 'web_scraper'
                
                action = Action(
                    action_name=tool_name,
                    description=data.get('desc', '') or data.get('title', ''),
                    order=order_counter,
                    args=data.get('tool_parameters', {})
                )
                self.utr.actions.append(action)
                order_counter += 1
                
            elif node_type == 'llm':
                data = node.get('data', {})
                model_name = data.get('model', {}).get('name', 'llm')
                # Use generic name for LLM to better match with generated actions
                action_name = "llm_generation"
                
                action = Action(
                    action_name=action_name,
                    description=data.get('title', 'LLM Process'),
                    order=order_counter,
                    args={
                        "prompt_template": data.get('prompt_template', []),
                        "model_config": data.get('model', {}),
                        "specific_model": model_name
                    }
                )
                self.utr.actions.append(action)
                order_counter += 1
                
            elif node_type == 'code':
                data = node.get('data', {})
                action = Action(
                    action_name="code_execution",
                    description=data.get('title', 'Run Code'),
                    order=order_counter,
                    args={
                        "code": data.get('code', ''),
                        "language": data.get('code_language', 'python')
                    }
                )
                self.utr.actions.append(action)
                order_counter += 1

            elif node_type == 'http-request':
                data = node.get('data', {})
                action = Action(
                    action_name="http_request",
                    description=data.get('title', 'HTTP Request'),
                    order=order_counter,
                    args={
                        "url": data.get('url', ''),
                        "method": data.get('method', 'GET'),
                        "headers": data.get('headers', ''),
                        "body": data.get('body', '')
                    }
                )
                self.utr.actions.append(action)
                order_counter += 1

    def _parse_control_intents(self):
        # Infer control flow from edges
        # Simple rule: if a node has multiple outgoing edges, it might be a branch
        # Dify DSL handles branching via 'if-else' nodes usually, or just linear flow
        # We will look for explicit 'if-else' or 'question-classifier' nodes
        
        for node_id, node in self.nodes.items():
            node_type = node.get('data', {}).get('type')
            
            if node_type == 'if-else':
                # Logic for if-else
                pass
            elif node_type == 'question-classifier':
                # Logic for classifier
                pass

    def _parse_resources(self):
        # Extract unique tool names as resources
        tools = set()
        for action in self.utr.actions:
            if action.action_name not in ['run_code', 'http_request'] and not action.action_name.startswith('llm_'):
                tools.add(action.action_name)
                
        for tool in tools:
            self.utr.resources.append(Resource(
                name=tool,
                type=ResourceType.tool,
                endpoint=None
            ))
