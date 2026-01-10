import os
import psutil

def measure_fit_ram_mb(model, X_train, y_train):
    process = psutil.Process(os.getpid())

    mem_before = process.memory_info().rss / 1024**2

    model.fit(X_train, y_train)

    mem_after = process.memory_info().rss / 1024**2

    return mem_after - mem_before