"""
System resource management utilities.

This module provides utilities for managing system resources, particularly
thread limits for scientific computing libraries.
"""
import os
import multiprocessing


def setup_thread_limits():
    """
    Set thread limits based on system load and available CPU resources.
    
    This function dynamically adjusts thread limits for scientific computing
    libraries (NumPy, MKL, OpenBLAS, etc.) based on current system load to
    prevent resource contention and deadlocks on overloaded systems.
    
    Strategy:
    - If system load is extremely high (load average > 2x CPU cores), limit to 1 thread
    - If system load is high (load average > CPU cores but <= 2x CPU cores), use 1 thread
    - If system load is normal (load average <= CPU cores), use 75% of available CPU resources
    - Thread count is capped at a reasonable maximum (64) to prevent over-allocation on large servers
    - Can be overridden by environment variables (if OMP_NUM_THREADS is already set)
    
    Environment variables set:
    - OMP_NUM_THREADS: OpenMP threads (used by many scientific libraries)
    - MKL_NUM_THREADS: Intel MKL threads
    - NUMEXPR_NUM_THREADS: NumExpr threads
    - OPENBLAS_NUM_THREADS: OpenBLAS threads
    - VECLIB_MAXIMUM_THREADS: macOS Accelerate framework threads
    - NUMBA_NUM_THREADS: Numba threads
    
    Examples
    --------
    >>> # Call before importing numpy or other libraries that use threading
    >>> from utils.system import setup_thread_limits
    >>> setup_thread_limits()
    >>> import numpy as np  # Now numpy will respect the thread limits
    """
    # Check if user has explicitly set thread limits via environment variable
    if 'OMP_NUM_THREADS' in os.environ:
        return  # User has explicit preference, don't override
    
    try:
        import psutil
        cpu_count = multiprocessing.cpu_count()
        load_avg = os.getloadavg()[0]  # 1-minute load average
        
        # Helper function to set all thread-related environment variables
        def set_thread_env(thread_count):
            os.environ['OMP_NUM_THREADS'] = str(thread_count)
            os.environ['MKL_NUM_THREADS'] = str(thread_count)
            os.environ['NUMEXPR_NUM_THREADS'] = str(thread_count)
            os.environ['OPENBLAS_NUM_THREADS'] = str(thread_count)
            os.environ['VECLIB_MAXIMUM_THREADS'] = str(thread_count)  # macOS
            os.environ['NUMBA_NUM_THREADS'] = str(thread_count)  # Numba
        
        # If load is more than 2x CPU cores, system is extremely overloaded
        if load_avg > 2 * cpu_count:
            print(f"[Thread Control] System load extremely high (load={load_avg:.1f} > {2*cpu_count}), "
                  f"limiting threads to 1 to avoid resource contention")
            set_thread_env(1)
        elif load_avg > cpu_count:
            # High load (load > cpu_count but <= 2*cpu_count): use very conservative thread limit
            # When system is already busy, use only 1 thread to avoid contention and deadlock
            max_threads = 1  # Force single-threaded when system is busy
            print(f"[Thread Control] System load high (load={load_avg:.1f} > {cpu_count}), "
                  f"forcing single-threaded mode to avoid resource contention")
            set_thread_env(max_threads)
        else:
            # Normal load (load <= cpu_count): calculate available CPU resources
            cpu_percent = psutil.cpu_percent(interval=0.1)  # Quick check of CPU usage (returns float)
            cpu_percent_float = float(cpu_percent) if not isinstance(cpu_percent, (list, tuple)) else cpu_percent[0]
            available_cpu_ratio = 1.0 - (cpu_percent_float / 100.0)
            available_cpus = max(1, int(cpu_count * available_cpu_ratio))
            
            # Use 75% of available CPUs, but ensure at least 1 thread
            max_threads = max(1, int(available_cpus * 0.75))
            # Cap at 64 threads to prevent over-allocation on very large servers (e.g., 512 CPUs)
            max_threads = min(max_threads, 64)
            
            print(f"[Thread Control] System load normal (load={load_avg:.1f}), "
                  f"CPU usage: {cpu_percent_float:.1f}%, "
                  f"Available CPUs: {available_cpus}/{cpu_count}, "
                  f"Setting threads to {max_threads} (75% of available)")
            set_thread_env(max_threads)
    except (ImportError, OSError):
        # psutil not available or can't get load average (e.g., on Windows)
        # Use conservative default: limit to 2 threads
        print("[Thread Control] Cannot detect system load, using conservative default (2 threads)")
        os.environ['OMP_NUM_THREADS'] = '2'
        os.environ['MKL_NUM_THREADS'] = '2'
        os.environ['NUMEXPR_NUM_THREADS'] = '2'
        os.environ['OPENBLAS_NUM_THREADS'] = '2'
        os.environ['VECLIB_MAXIMUM_THREADS'] = '2'
        os.environ['NUMBA_NUM_THREADS'] = '2'
