import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Any, Optional, Union, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrogradeAnalysis:
    """Main class for performing retrograde analysis on deep learning models."""
    
    def __init__(self, model: Union[nn.Module, str], config: Optional[Dict[str, Any]] = None):
        """
        Initialize the retrograde analysis tool.
        
        Args:
            model: Either a PyTorch model or a string identifier for a Hugging Face model
            config: Optional configuration dictionary for custom models
        """
        self.model = self._load_model(model, config)
        self.transformation_log = []
        self.metadata = None
        self.architecture = None
        self.layer_analysis = {}
        
    def _load_model(self, model: Union[nn.Module, str], config: Optional[Dict[str, Any]]) -> nn.Module:
        """Load model from various sources."""
        if isinstance(model, str):
            try:
                # Try to load using AutoModel
                if config is not None:
                    hf_config = AutoConfig.from_dict(config)
                    return AutoModel.from_pretrained(model, config=hf_config)
                else:
                    return AutoModel.from_pretrained(model)
            except Exception as e:
                logger.error(f"Error loading Hugging Face model: {e}")
                raise
        elif isinstance(model, nn.Module):
            return model
        else:
            raise ValueError("Model must be either a PyTorch nn.Module or a Hugging Face model identifier")
    
    def extract_metadata(self) -> Dict[str, Any]:
        """Extract model metadata including architecture details and parameters."""
        try:
            metadata = {
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                'layer_count': sum(1 for _ in self.model.modules()),
                'device': next(self.model.parameters()).device
            }
        except StopIteration:
            metadata = {
                'total_parameters': 0,
                'trainable_parameters': 0,
                'layer_count': 0,
                'device': 'cpu'
            }
        
        if hasattr(self.model, 'config'):
            # Handle Hugging Face models
            config_dict = vars(self.model.config)
            metadata.update({
                'model_type': config_dict.get('model_type', 'unknown'),
                'hidden_size': config_dict.get('hidden_size'),
                'num_attention_heads': config_dict.get('num_attention_heads'),
                'num_hidden_layers': config_dict.get('num_hidden_layers')
            })
        else:
            metadata['model_type'] = type(self.model).__name__
            
        self.metadata = metadata
        return metadata
    
    def analyze_layer(self, layer: nn.Module) -> Dict[str, Any]:
        """Perform detailed analysis of a single layer."""
        layer_info = {
            'type': type(layer).__name__,
            'parameters': sum(p.numel() for p in layer.parameters())
        }
        
        # Weight analysis for different layer types
        if isinstance(layer, nn.Linear):
            if hasattr(layer, 'weight') and layer.weight is not None:
                weights = layer.weight.data
                layer_info.update({
                    'in_features': layer.in_features,
                    'out_features': layer.out_features,
                    'bias': layer.bias is not None,
                    'weight_stats': self._compute_weight_statistics(weights),
                    'singular_values': self._compute_svd(weights)
                })
                
        elif isinstance(layer, nn.Conv2d):
            if hasattr(layer, 'weight') and layer.weight is not None:
                weights = layer.weight.data
                layer_info.update({
                    'in_channels': layer.in_channels,
                    'out_channels': layer.out_channels,
                    'kernel_size': layer.kernel_size,
                    'stride': layer.stride,
                    'padding': layer.padding,
                    'dilation': layer.dilation,
                    'groups': layer.groups,
                    'bias': layer.bias is not None,
                    'weight_stats': self._compute_weight_statistics(weights),
                    'kernel_analysis': self._analyze_conv_kernels(weights)
                })
                
        elif isinstance(layer, (nn.LSTM, nn.GRU)):
            layer_info.update({
                'input_size': layer.input_size,
                'hidden_size': layer.hidden_size,
                'num_layers': layer.num_layers,
                'bias': layer.bias,
                'batch_first': layer.batch_first,
                'bidirectional': layer.bidirectional
            })
            # Analyze weights for RNN layers
            weights = []
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    weights.append(param.data)
            layer_info['weight_stats'] = [self._compute_weight_statistics(w) for w in weights]
            
        elif isinstance(layer, (nn.ReLU, nn.GELU, nn.Tanh, nn.Sigmoid, nn.Softmax, nn.LeakyReLU, nn.ELU)):
            layer_info['activation_type'] = type(layer).__name__
        
        elif isinstance(layer, nn.Embedding):
            if hasattr(layer, 'weight') and layer.weight is not None:
                weights = layer.weight.data
                layer_info.update({
                    'num_embeddings': layer.num_embeddings,
                    'embedding_dim': layer.embedding_dim,
                    'weight_stats': self._compute_weight_statistics(weights)
                })
        
        elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if hasattr(layer, 'weight') and layer.weight is not None:
                weights = layer.weight.data
                layer_info.update({
                    'num_features': layer.num_features,
                    'eps': layer.eps,
                    'momentum': layer.momentum,
                    'affine': layer.affine,
                    'track_running_stats': layer.track_running_stats,
                    'weight_stats': self._compute_weight_statistics(weights)
                })
        
        elif isinstance(layer, nn.LayerNorm):
            if hasattr(layer, 'weight') and layer.weight is not None:
                weights = layer.weight.data
                layer_info.update({
                    'normalized_shape': layer.normalized_shape,
                    'eps': layer.eps,
                    'elementwise_affine': layer.elementwise_affine,
                    'weight_stats': self._compute_weight_statistics(weights)
                })
        
        elif isinstance(layer, nn.MultiheadAttention):
            layer_info.update({
                'embed_dim': layer.embed_dim,
                'num_heads': layer.num_heads,
                'kdim': layer.kdim,
                'vdim': layer.vdim,
                'batch_first': layer.batch_first
            })
            # Analyze the in-projection weights
            if hasattr(layer, 'in_proj_weight') and layer.in_proj_weight is not None:
                weights = layer.in_proj_weight.data
                layer_info['weight_stats'] = self._compute_weight_statistics(weights)
        
        else:
            # For other layer types, attempt to analyze weights if they exist
            if hasattr(layer, 'weight') and layer.weight is not None:
                weights = layer.weight.data
                layer_info.update({
                    'weight_stats': self._compute_weight_statistics(weights)
                })
        
        return layer_info
    
    def _compute_weight_statistics(self, weights: torch.Tensor) -> Dict[str, float]:
        """Compute statistical measures for weight tensors."""
        with torch.no_grad():
            weights_np = weights.cpu().numpy()
            return {
                'mean': float(np.mean(weights_np)),
                'std': float(np.std(weights_np)),
                'min': float(np.min(weights_np)),
                'max': float(np.max(weights_np)),
                'sparsity': float(np.count_nonzero(weights_np == 0) / weights_np.size)
            }
    
    def _compute_svd(self, weights: torch.Tensor) -> Dict[str, Any]:
        """Compute Singular Value Decomposition of weight matrix."""
        with torch.no_grad():
            if len(weights.shape) > 2:
                weights = weights.reshape(weights.shape[0], -1)
            try:
                u, s, v = torch.svd(weights)
                singular_values = s.cpu().numpy()
                condition_number = float(singular_values[0] / singular_values[-1]) if s[-1] != 0 else float('inf')
                return {
                    'singular_values': singular_values,
                    'condition_number': condition_number
                }
            except RuntimeError as e:
                logger.warning(f"SVD computation failed for weights of shape {weights.shape}: {e}")
                return {
                    'singular_values': None,
                    'condition_number': None
                }
    
    def _analyze_conv_kernels(self, weights: torch.Tensor) -> Dict[str, Any]:
        """Analyze convolutional kernels."""
        with torch.no_grad():
            weights_np = weights.cpu().numpy()
            num_kernels = weights_np.shape[0]
            kernels_flat = weights_np.reshape(num_kernels, -1)
            kernel_norms = np.linalg.norm(kernels_flat, axis=1)
            kernel_correlations = np.corrcoef(kernels_flat)
            return {
                'kernel_norms': kernel_norms,
                'kernel_correlations': kernel_correlations
            }
    
    def parse_architecture(self) -> Dict[str, Any]:
        """Parse and analyze the complete model architecture."""
        architecture = {}
        edges = []
        node_ids = {}
        node_counter = 0
        
        for name, module in self.model.named_modules():
            # Assign a unique ID to each module to avoid name conflicts
            node_id = f"{name}_{node_counter}"
            node_ids[name] = node_id
            node_counter += 1
            if not name:  # Skip root module
                continue
                
            parent_name = '.'.join(name.split('.')[:-1])
            if parent_name and parent_name in node_ids:
                edges.append((node_ids[parent_name], node_id))
                
            # Analyze the layer
            layer_info = self.analyze_layer(module)
            architecture[node_id] = {
                'name': name,
                'type': type(module).__name__,
                'parameters': sum(p.numel() for p in module.parameters()),
                'layer_info': layer_info
            }
                
        self.architecture = {
            'nodes': architecture,
            'edges': edges
        }
        return self.architecture
    
    def simplify_architecture(self, threshold: float = 1e-3) -> nn.Module:
        """Simplify the model architecture through pruning and optimization."""
        logger.info(f"Starting model simplification with threshold {threshold}")
        
        simplified_model = self.model
        total_params_before = sum(p.numel() for p in simplified_model.parameters())
        
        # Prune small weights
        with torch.no_grad():
            for name, module in simplified_model.named_modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    mask = torch.abs(module.weight.data) > threshold
                    elements_removed = int((~mask).sum().item())
                    module.weight.data *= mask
                    self.transformation_log.append({
                        'layer': name,
                        'operation': 'weight_pruning',
                        'threshold': threshold,
                        'elements_removed': elements_removed
                    })
                    logger.debug(f"Pruned {elements_removed} elements in layer {name}")
    
        total_params_after = sum(p.numel() for p in simplified_model.parameters())
        reduction = (total_params_before - total_params_after) / total_params_before * 100
        
        logger.info(f"Model simplified: {reduction:.2f}% parameter reduction")
        return simplified_model
    
    def visualize_architecture(self, output_path: Optional[str] = None):
        """Generate a visualization of the model architecture."""
        if not self.architecture:
            self.parse_architecture()
            
        G = nx.DiGraph()
        
        # Add nodes
        for node_id, info in self.architecture['nodes'].items():
            G.add_node(node_id, label=info['name'], type=info['type'])
            
        # Add edges
        G.add_edges_from(self.architecture['edges'])
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(G)
        node_types = [G.nodes[node]['type'] for node in G.nodes()]
        unique_types = list(set(node_types))
        type_colors = {t: plt.cm.tab20(i) for i, t in enumerate(unique_types)}
        node_colors = [type_colors[t] for t in node_types]
        
        nx.draw(G, pos, with_labels=True, labels={node: G.nodes[node]['label'] for node in G.nodes()},
                node_color=node_colors, node_size=1000, arrowsize=20, font_size=8)
        
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Architecture visualization saved to {output_path}")
        else:
            plt.show()
        plt.close()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive analysis report."""
        if not self.metadata:
            self.extract_metadata()
        if not self.architecture:
            self.parse_architecture()
            
        report = {
            'metadata': self.metadata,
            'architecture_summary': {
                'total_layers': len(self.architecture['nodes']),
                'layer_types': defaultdict(int)
            },
            'weight_statistics': defaultdict(list),
            'transformation_log': self.transformation_log
        }
        
        # Collect layer type statistics
        for node_info in self.architecture['nodes'].values():
            report['architecture_summary']['layer_types'][node_info['type']] += 1
            
        # Collect weight statistics
        for node_info in self.architecture['nodes'].values():
            layer_info = node_info['layer_info']
            if 'weight_stats' in layer_info:
                for stat_name, value in layer_info['weight_stats'].items():
                    report['weight_statistics'][stat_name].append(value)
                        
        return report

def example_usage():
    """Example usage of the RetrogradeAnalysis class."""
    # Initialize with a Hugging Face model
    analyzer = RetrogradeAnalysis('bert-base-uncased')
    
    # Extract metadata
    metadata = analyzer.extract_metadata()
    print("Model Metadata:", metadata)
    
    # Parse architecture
    architecture = analyzer.parse_architecture()
    print("\nArchitecture Analysis Complete")
    
    # Simplify model
    simplified_model = analyzer.simplify_architecture(threshold=1e-3)
    print("\nModel Simplification Complete")
    
    # Generate visualization
    analyzer.visualize_architecture('model_architecture.png')
    print("\nVisualization saved as 'model_architecture.png'")
    
    # Generate report
    report = analyzer.generate_report()
    print("\nAnalysis Report:", report)

if __name__ == "__main__":
    example_usage()
