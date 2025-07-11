"""
Core server logic for the IDP-FL algorithm.
"""

import torch
from typing import List, Dict
from tqdm import tqdm


class Server():
    """
    This server implements:
    1. privacy-aware client selection
    2. convergence-generalization-driven batch size optimization
    3. incentive mechanism
    4. model aggregation
    """
    
    def __init__(self, model, config, test_loader):
        """
        Initialize the IDP-FL server.
        
        Args:
            model: Global model architecture
            config (dict): Configuration parameters
            test_loader: Test data loader for evaluation
        """
        self.model = model
        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model = self.model.to(self.device)

        # Incentive mechanism parameters
        self.a_k = config['profit_adjustment_factor_a']
        self.b_k = config['profit_adjustment_factor_b']
        self.theta = config['payment_adjustment_parameter']
        self.utility_factor = config['utility_factor']

    def compute_optimal_payment(self, client, B_k):
        """
        Compute optimal payment for client based on incentive mechanism.
           
        Args:
            client: Client instance
            B_k: Batch size for client k
            
        Returns:
            float: Optimal payment amount
        """
        pass

    def select_clients(self, all_clients: List) -> List:
        """
        Privacy-aware client selection
        
        Args:
            all_clients (List): All available clients
            
        Returns:
            List: Selected clients for current round
        """
        pass


    def aggregate(self, client_updates: Dict[int, torch.Tensor]) -> None:
        """
        Model aggregation
        
        Args:
            client_updates (Dict[int, torch.Tensor]): Client model updates
        """
        total_data_sizes = sum(len(client.subset) for client in self.selected_clients)
        aggregated_model = {k: torch.zeros_like(v).to(self.device) 
                           for k, v in self.global_model.state_dict().items()}

        for client_id, update in client_updates.items():
            client = next(c for c in self.selected_clients if c.client_id == client_id)
            weight = len(client.subset) / total_data_sizes
            for key in aggregated_model:
                aggregated_model[key] += weight * update[key].to(self.device)

        self.global_model.load_state_dict(aggregated_model)

    def evaluate(self) -> float:
        """
        Evaluate the global model on test dataset.
        
        Returns:
            float: Test accuracy
        """
        self.global_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to(self.device)
                outputs = self.global_model(data)
                _, predicted = torch.max(outputs.data, 1)
                
                if isinstance(target, torch.Tensor):
                    target = target.to(self.device)
                    total += target.numel()
                    correct += (predicted == target).sum().item()
                else:
                    total += 1
                    correct += (predicted.item() == target)
                    
        return correct / total

    def train(self, clients: List, num_rounds: int) -> None:
        """
        Main training loop for IDP-FL.
        
        Args:
            clients (List): All available clients
            num_rounds (int): Number of training rounds
        """
        for round in tqdm(range(num_rounds), desc="Training Progress"):
            # Phase 1: Client Selection
            self.selected_clients = self.select_clients(clients)
            
            # Phase 2: Batch Size Computation
            B_values = self.compute_batch_sizes()
            
            # Phase 3: Client Training and Incentive Computation
            self.client_updates = {}
            client_utilities = {}
            client_lcts = []
            client_energies = []

            for client, B_k in zip(self.selected_clients, B_values):
                # Compute optimal payment
                g_k = self.compute_optimal_payment(client, B_k)

                # Distribute global model to client
                client.model.load_state_dict(self.global_model.state_dict())

                # Client training with differential privacy
                client_update, f_k = client.train(B_k, round+1, g_k, self.theta)
                self.client_updates[client.client_id] = client_update

                # TODO Compute utilities for incentive mechanism
                

            # Phase 4: Model Aggregation
            self.aggregate(self.client_updates)

    def compute_batch_sizes(self):
        """
        Compute optimal batch sizes for selected clients.
        
        Returns:
            List[int]: Optimal batch sizes for each client
        """
        pass