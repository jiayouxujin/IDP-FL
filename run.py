"""
Main execution script for the IDP-FL algorithm.
"""

import math
import random
from algorithms.idp_fl.server import Server
from algorithms.idp_fl.client import Client



def IDP_FL_Run(custom_config, shared_clients):
    """
    Main execution function for IDP-FL algorithm.
    
    Args:
        custom_config (dict): Custom configuration parameters
        shared_clients (list): List of client configurations
    
    Returns:
        float: Final test accuracy
    """
    # Base configuration
    config = {
        'experiment_name': 'idp_fl_experiment',
        # Core algorithm parameters
        'num_rounds': 100,
        'num_selected_clients': 10,
        'learning_rate': 0.01,
        'local_epochs': 5,
        
        # Differential privacy parameters including delta, max_grad_norm
        
        # Incentive mechanism parameters including profit_adjustment_factor_a, profit_adjustment_factor_b, payment_adjustment_parameter, utility_factor
        
        # Client selection weights including weight_non_iid, weight_dp, alpha
        
        # Resource parameters including cpu_cycles_per_sample, min_cpu_freq, max_cpu_freq, energy_efficiency
    }

    # Update with custom configuration if provided
    if custom_config:
        config.update(custom_config)
        config['log_file'] = f"{config['experiment_name']}_{config['log_file']}"

    # Load dataset
    print("Loading Fashion-MNIST dataset...")
    _, test_dataset = load_and_preprocess_fmnist('./data')

    # Initialize global model
    print("Initializing global model...")
    model = LeNet()

    # Create server instance
    print("Creating server...")
    server = Server(model, config, test_dataset)

    # Create client instances
    print("Creating clients...")
    clients = []
    for client_data in shared_clients:
        client = Client(
            client_data['client_id'],
            client_data['subset'],
            client_data['model'],
            client_data['num_labels'],
            client_data['non_iid_degree'],
            client_data['epsilon'],
            config
        )
        clients.append(client)

    # Normalize client parameters
    print("Normalizing client parameters...")
    # Normalize non-IID degree (1-max normalization)
    max_non_iid_degree = max(client.non_iid_degree for client in clients)
    for client in clients:
        client.non_iid_degree = 1 - ((client.non_iid_degree) / (max_non_iid_degree))

    # Normalize epsilon (max normalization)
    max_epsilon = max(client.epsilon for client in clients)
    for client in clients:
        client.normalize_epsilon = client.epsilon / max_epsilon

    # Start federated learning training
    print("Starting IDP-FL training...")
    server.train(clients, config['num_rounds'])

    # Evaluate final model
    print("Evaluating final model...")
    accuracy = server.evaluate()
    print(f"Final Test Accuracy: {accuracy:.4f}")
    
    return accuracy
