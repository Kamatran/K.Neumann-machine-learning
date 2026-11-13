from problems import (
    load_problem, fit_and_log, standardized, search_alpha, plot_search, plot_path,
    RIDGE_ALPHAS, LASSO_ALPHAS, LASSO_MAX_ITER
)
from weights import choose_alpha, weight_summary, top_weights
from sklearn.linear_model import Ridge, Lasso
import mlflow
from mlproject.tracking import setup
from problems import mse

def run_regularization(problem_name, model_name, make_model, alphas):
    data = load_problem(problem_name)
    setup(f"w03-{problem_name}")
    
    search, path = search_alpha(make_model, alphas, data)
    alpha = choose_alpha(search)
    
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("alpha", alpha)
        
        mlflow.log_table(search, "alpha_search.json")
        
        fig_search = plot_search(search, alpha)
        mlflow.log_figure(fig_search, "search.png")
        
        fig_path = plot_path(path, alpha)
        mlflow.log_figure(fig_path, "path.png")
        
        model, w = fit_and_log(standardized(make_model(alpha)), data)
        
        if w is not None:
            w_summary = weight_summary(w)
            top_w = top_weights(w, k=10)
            mlflow.log_metrics(w_summary)
            mlflow.log_dict(top_w, "top_weights.json")


def run_test_once(problem_name):
    data = load_problem(problem_name)
    setup(f"w03-{problem_name}")
    
    search_ridge, _ = search_alpha(lambda a: Ridge(alpha=a), RIDGE_ALPHAS, data)
    alpha_ridge = choose_alpha(search_ridge)
    
    search_lasso, _ = search_alpha(
        lambda a: Lasso(alpha=a, max_iter=LASSO_MAX_ITER), 
        LASSO_ALPHAS, 
        data
    )
    alpha_lasso = choose_alpha(search_lasso)
    
    with mlflow.start_run(run_name="test-once"):
        # OLS
        from sklearn.linear_model import LinearRegression
        ols_model, _ = fit_and_log(standardized(LinearRegression()), data)
        test_mse_ols = mse(data.y_test, ols_model.predict(data.X_test))
        mlflow.log_metric("test_mse_ols", test_mse_ols)
        
        # Ridge
        ridge_model, _ = fit_and_log(standardized(Ridge(alpha=alpha_ridge)), data)
        test_mse_ridge = mse(data.y_test, ridge_model.predict(data.X_test))
        mlflow.log_metric("test_mse_ridge", test_mse_ridge)
        
        # Lasso
        lasso_model, _ = fit_and_log(
            standardized(Lasso(alpha=alpha_lasso, max_iter=LASSO_MAX_ITER)), 
            data
        )
        test_mse_lasso = mse(data.y_test, lasso_model.predict(data.X_test))
        mlflow.log_metric("test_mse_lasso", test_mse_lasso)
        
        # Tag
        mlflow.set_tag("test_evaluated", "true")


if __name__ == "__main__":
    for problem in ["noise", "cholesterol"]:
        # Ridge
        run_regularization(
            problem, 
            "ridge", 
            lambda a: Ridge(alpha=a), 
            RIDGE_ALPHAS
        )
        
        # Lasso
        run_regularization(
            problem, 
            "lasso", 
            lambda a: Lasso(alpha=a, max_iter=LASSO_MAX_ITER), 
            LASSO_ALPHAS
        )
        
        # Krok 3: test-once
        run_test_once(problem)