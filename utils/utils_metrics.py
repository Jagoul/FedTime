import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    ) / torch.sum(eval_points)


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j: j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS_sum(target, forecast, eval_points, mean_scaler, scaler):
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast, quantiles[i], dim=1)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


# Federated learning specific metrics
def federated_average_metric(client_metrics, client_weights=None):
    """
    Calculate federated average of metrics across clients
    
    Args:
        client_metrics: List of metric values from each client
        client_weights: Optional weights for each client
    
    Returns:
        Weighted average of metrics
    """
    if client_weights is None:
        client_weights = [1.0] * len(client_metrics)
    
    # Normalize weights
    total_weight = sum(client_weights)
    client_weights = [w / total_weight for w in client_weights]
    
    # Calculate weighted average
    fed_metric = sum(metric * weight for metric, weight in zip(client_metrics, client_weights))
    return fed_metric


def communication_efficiency_metric(model_size_bytes, num_rounds, num_clients):
    """
    Calculate communication efficiency metric
    
    Args:
        model_size_bytes: Size of model in bytes
        num_rounds: Number of federated rounds
        num_clients: Number of clients
    
    Returns:
        Communication efficiency score
    """
    total_communication = model_size_bytes * num_rounds * num_clients * 2  # Upload + download
    # Convert to MB and return efficiency score (lower is better)
    return total_communication / (1024 * 1024)


def privacy_preservation_score(noise_multiplier, max_grad_norm):
    """
    Calculate privacy preservation score based on differential privacy parameters
    
    Args:
        noise_multiplier: Noise multiplier for DP
        max_grad_norm: Maximum gradient norm for clipping
    
    Returns:
        Privacy score (higher is better privacy)
    """
    if noise_multiplier == 0:
        return 0.0
    return min(noise_multiplier / max_grad_norm, 10.0)  # Cap at 10 for normalization


def federated_fairness_metric(client_performances):
    """
    Calculate fairness metric across federated clients
    
    Args:
        client_performances: List of performance scores for each client
    
    Returns:
        Fairness score (lower variance indicates higher fairness)
    """
    return np.var(client_performances)


def convergence_rate_metric(loss_history, threshold=0.01):
    """
    Calculate convergence rate based on loss history
    
    Args:
        loss_history: List of loss values over training
        threshold: Convergence threshold
    
    Returns:
        Number of epochs to converge
    """
    if len(loss_history) < 2:
        return len(loss_history)
    
    for i in range(1, len(loss_history)):
        if abs(loss_history[i] - loss_history[i-1]) < threshold:
            return i
    
    return len(loss_history)
