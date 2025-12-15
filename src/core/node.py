import numpy as np

class CorticalColumn:
    def __init__(self, order=2, dt=0.01, learning_rate=1.0, w_learning_rate=0.01):
        """
        A Generalized Cortical Column with Hierarchical Support.
        """
        self.order = order
        self.dt = dt
        self.k = learning_rate
        self.eta = w_learning_rate
        
        self.mu = np.zeros(order + 1)
        self.W = np.zeros((order + 1, order + 1))
        
        self.prev_input = None
        self.prev_input_vel = 0.0
        
        # Errors
        self.ez = np.zeros(order + 1)      # Sensory Error (Bottom-Up)
        self.ez_prior = np.zeros(order + 1) # Prediction Error (Top-Down)

    def predict_motion(self):
        d_mu_inertial = np.zeros_like(self.mu)
        for i in range(self.order):
            d_mu_inertial[i] = self.mu[i+1]
        
        d_mu_learned = self.W @ self.mu
        return d_mu_inertial + d_mu_learned

    def estimate_input_derivatives(self, input_value):
        y = np.zeros(self.order + 1)
        y[0] = input_value
        if self.prev_input is None:
            self.prev_input = input_value
            return y
            
        velocity = (input_value - self.prev_input) / self.dt
        y[1] = velocity
        
        acceleration = (velocity - self.prev_input_vel) / self.dt
        if self.order >= 2:
            y[2] = acceleration
            
        self.prev_input = input_value
        self.prev_input_vel = velocity
        return y

    def step(self, input_value, input_precision=1.0, top_down_prediction=None, top_down_precision=0.0):
        """
        Active Inference Step with Hierarchical Inputs.
        
        Args:
            input_value: Sensory data (or input from lower layer).
            input_precision: Confidence in sensory data.
            top_down_prediction (np.array): Expected state from parent layer.
            top_down_precision (float): Confidence in parent's prediction.
        """
        # 1. Estimate Generalized Input (Bottom-Up Target)
        if input_precision > 0:
            target_y = self.estimate_input_derivatives(input_value)
        else:
            target_y = self.mu.copy()

        # 2. Intrinsic Prediction (Dynamics)
        total_derivative = self.predict_motion()
        
        # 3. Compute Errors
        
        # A. Sensory Error (Prediction Error sent UP)
        # e_z = y - mu
        sensory_error = target_y - self.mu
        weighted_sensory_error = sensory_error * input_precision
        self.ez = weighted_sensory_error
        
        # B. Prior Error (Prediction Error sent DOWN from Parent)
        # We are the Child. Parent says we should be at `top_down_prediction`.
        # e_prior = prior - mu
        if top_down_prediction is not None:
            # Parent usually predicts our Value (mu[0]), maybe Vel?
            # Assume parent predicts full state vector for simplicity, or just mu[0].
            # If parent is a higher abstract node, it might only predict mu[0].
            # Let's align dimensions.
            prior_error = top_down_prediction - self.mu
            weighted_prior_error = prior_error * top_down_precision
        else:
            weighted_prior_error = np.zeros_like(self.mu)
            
        self.ez_prior = weighted_prior_error

        # 4. State Update (Inference)
        # dot_mu = D*mu + k_sensory * e_z + k_prior * e_prior
        # We are pulled by Reality (Input) AND by Expectation (Parent).
        
        state_update = total_derivative + \
                       self.k * weighted_sensory_error + \
                       self.k * weighted_prior_error
        
        old_mu = self.mu.copy()
        
        # Integration
        self.mu += state_update * self.dt
        
        # 5. Weight Update (Lateral / Internal Learning)
        # Learn intrinsic dynamics to minimize TOTAL error
        # dW = eta * (Error * State^T)
        # We use the net error direction
        net_error = weighted_sensory_error + weighted_prior_error
        
        if input_precision > 0 or top_down_precision > 0:
            dW = self.eta * np.outer(net_error, old_mu)
            self.W += dW * self.dt
            self.W -= 0.01 * self.W * self.dt # Decay

        return self.mu.copy() # Return full state for hierarchy