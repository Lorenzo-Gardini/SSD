from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit

def grid_search(model, param_grid, train_x, train_y, n_splits=5):
    grid = GridSearchCV(model,
                        param_grid,
                        cv=TimeSeriesSplit(n_splits=n_splits),
                        verbose=True)
    grid.fit(train_x, train_y)
    print()
    print(f'Best params: {grid.best_params_}')
    print()
    return grid
