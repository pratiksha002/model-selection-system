def select_best_model(results):
    best_model = min(results, key=lambda x: results[x]["mae"])
    
    best_metrics = results[best_model]

    print(f"Best model: {best_model}")

    return best_model, best_metrics