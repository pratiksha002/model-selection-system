def select_best_model(results, metric="mae"):
    if metric in ["mae", "rmse"]:
        best_model = min(results, key=lambda x: results[x][metric])
    elif metric == "r2":
        best_model = max(results, key=lambda x: results[x][metric])
    else:
        raise ValueError("Invalid metric")
    
    best_metrics = results[best_model]

    print(f"Best model based on {metric}: {best_model}")

    return best_model, best_metrics