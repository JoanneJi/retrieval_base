#!/usr/bin/env python3
"""
Monitor resource usage of retrieval processes.

This script monitors CPU usage, memory consumption, thread count, and other
resource metrics for running Python scripts, particularly useful for monitoring
long-running retrieval processes.

Location:
    This script is located in the project root directory:
    analysis/retrieval/retrieval_base/monitor_retrieval.py
    
    The scripts to monitor are typically in the src/ subdirectory:
    analysis/retrieval/retrieval_base/src/

Usage Examples:
    # Basic usage: Monitor a Python script by filename
    python monitor_retrieval.py crires_retrieval_chips_starB.py
    # or with relative path:
    python monitor_retrieval.py src/crires_retrieval_chips_starB.py
    
    # Monitor with custom update interval (default is 2 seconds)
    python monitor_retrieval.py crires_retrieval_chips_starA.py --interval 5
    
    # Monitor by process ID directly
    python monitor_retrieval.py --pid 12345
    
    # Show detailed thread information
    python monitor_retrieval.py crires_retrieval_chips_starB.py --show-threads
    
    # List all matching processes without monitoring
    python monitor_retrieval.py crires_retrieval_chips_starB.py --list
    
    # Get help information
    python monitor_retrieval.py --help

Command Line Arguments:
    script_name (optional)
        Python script filename to monitor (e.g., crires_retrieval_chips_starB.py
        or src/crires_retrieval_chips_starB.py).
        The script will automatically find the running process matching this name
        in the command line, regardless of the full path.
    
    --pid PID
        Directly specify the process ID to monitor.
        If specified, script_name will be ignored.
    
    --interval INTERVAL
        Update interval in seconds (default: 2.0).
        Smaller values provide more frequent updates but use more CPU.
    
    --show-threads
        Display detailed thread information every 20 iterations.
        Shows thread IDs and counts.
    
    --list
        Only list matching processes without starting monitoring.
        Useful for finding the correct process ID when multiple instances are running.

Output Information:
    The script displays real-time information including:
    - Time: Current timestamp
    - CPU%: Process CPU usage percentage
    - Memory(MB/GB): Process memory consumption
    - Threads: Number of threads in the process
    - Children: Number of child processes
    - Sys CPU%: System-wide CPU usage
    - Sys Mem%: System-wide memory usage
    
    Every 20 iterations, additional statistics are shown:
    - Maximum CPU usage reached
    - Maximum memory usage reached
    - Thread details (if --show-threads is enabled)
    - Detailed memory breakdown (RSS, VMS, Shared)

Requirements:
    - psutil library: pip install psutil
    - Python 3.6 or higher

Author:
    Chenyang Ji (2025-01-22)
"""
import psutil
import sys
import os
import time
import argparse
from pathlib import Path


def find_process_by_script_name(script_name):
    """
    Find process by Python script name.
    
    Parameters
    ----------
    script_name : str
        Python script filename, e.g., 'crires_retrieval_chips_starB.py'
    
    Returns
    -------
    psutil.Process or None
        Found process object, or None if not found
    """
    script_name = script_name.strip()
    if not script_name.endswith('.py'):
        script_name = script_name + '.py'
    
    # Get current process ID to exclude self
    current_pid = os.getpid()
    
    # Extract just the filename from path (e.g., 'src/crires_retrieval_chips_starB.py' -> 'crires_retrieval_chips_starB.py')
    script_basename = os.path.basename(script_name)
    
    matching_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'username']):
        try:
            # Skip current process (monitor_retrieval.py itself)
            if proc.pid == current_pid:
                continue
            
            cmdline = proc.info['cmdline']
            if cmdline is None or len(cmdline) == 0:
                continue
            
            # Check if it's a Python process
            if 'python' not in proc.info['name'].lower():
                continue
            
            # Check if command line contains target script name
            # The script should be the main script being run, not an argument to another script
            cmdline_str = ' '.join(cmdline)
            
            # Check if the script name appears in the command line
            if script_basename in cmdline_str:
                # Make sure it's not the monitor script itself
                # (monitor_retrieval.py's command line also contains the target script name as an argument)
                if 'monitor_retrieval.py' in cmdline_str:
                    continue  # Skip monitor_retrieval.py itself
                
                # Prefer processes where the script name appears earlier in the command line
                # (the main script is usually the second argument after 'python')
                if len(cmdline) > 1:
                    # Check if the script is the main script (second argument) or in the path
                    main_script = cmdline[1] if len(cmdline) > 1 else ''
                    if script_basename in main_script or script_basename in cmdline_str:
                        matching_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    
    if len(matching_processes) == 0:
        return None
    elif len(matching_processes) == 1:
        return matching_processes[0]
    else:
        # If multiple matching processes found, return the first one (usually the most recently started)
        print(f"Warning: Found {len(matching_processes)} matching processes, monitoring the first one (PID: {matching_processes[0].pid})")
        return matching_processes[0]


def format_bytes(bytes_value):
    """Format bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def monitor_process(process, interval=2, show_threads=False):
    """
    Monitor resource usage of a specified process.
    
    Parameters
    ----------
    process : psutil.Process
        Process object to monitor
    interval : float
        Update interval in seconds
    show_threads : bool
        Whether to show detailed thread information
    """
    pid = process.pid
    
    try:
        # Get process information
        cmdline = ' '.join(process.cmdline()[:5])  # Show only first 5 arguments
        if len(process.cmdline()) > 5:
            cmdline += '...'
        
        username = process.username()
        create_time = time.strftime('%Y-%m-%d %H:%M:%S', 
                                   time.localtime(process.create_time()))
        
        # Get system total resources
        cpu_count = psutil.cpu_count(logical=True)
        cpu_cores = psutil.cpu_count(logical=False)
        mem_total = psutil.virtual_memory().total / (1024**3)  # GB
        load_avg = os.getloadavg()
        
        print(f"\n{'='*90}")
        print(f"Monitoring Process: PID={pid}, User={username}, Start Time={create_time}")
        print(f"Command: {cmdline}")
        print(f"{'='*90}")
        print(f"System Resources: {cpu_count} logical CPUs ({cpu_cores} cores), "
              f"{mem_total:.1f} GB RAM, Load: {load_avg[0]:.2f} (1min)")
        print(f"{'='*90}")
        print(f"{'Time':<10} {'CPU%':<8} {'CPU Cores':<10} {'Memory(MB)':<12} {'Memory(GB)':<12} "
              f"{'Threads':<8} {'Children':<8} {'Sys CPU%':<10} {'Sys Mem%':<10}")
        print(f"{'-'*90}")
        
        # Initial call for CPU usage calculation
        process.cpu_percent()
        
        iteration = 0
        max_mem = 0
        max_cpu = 0
        
        try:
            while True:
                iteration += 1
                
                # CPU usage
                cpu_percent = process.cpu_percent(interval=interval)
                if cpu_percent > max_cpu:
                    max_cpu = cpu_percent
                
                # Calculate equivalent CPU cores used
                cpu_cores_used = cpu_percent / 100.0
                
                # Memory usage
                mem_info = process.memory_info()
                mem_mb = mem_info.rss / 1024 / 1024
                mem_gb = mem_mb / 1024
                if mem_mb > max_mem:
                    max_mem = mem_mb
                
                # Number of threads
                num_threads = process.num_threads()
                
                # Number of child processes
                try:
                    num_children = len(process.children(recursive=True))
                except:
                    num_children = 0
                
                # System resources
                sys_cpu = psutil.cpu_percent(interval=0.1)
                sys_mem = psutil.virtual_memory()
                
                current_time = time.strftime('%H:%M:%S')
                print(f"{current_time:<10} {cpu_percent:>6.1f}%  {cpu_cores_used:>8.1f}  {mem_mb:>10.0f}  {mem_gb:>10.2f}  "
                      f"{num_threads:>6}  {num_children:>6}  {sys_cpu:>8.1f}%  {sys_mem.percent:>8.1f}%")
                
                # Print detailed information and statistics every 20 iterations
                if iteration % 20 == 0:
                    max_cores_used = max_cpu / 100.0
                    cpu_usage_pct = (max_cpu / 100.0) / cpu_count * 100
                    mem_usage_pct = (max_mem / 1024) / mem_total * 100
                    print(f"\n  [Statistics] Max CPU usage: {max_cpu:.1f}% ({max_cores_used:.1f} cores, {cpu_usage_pct:.2f}% of system)")
                    print(f"  [Statistics] Max memory usage: {max_mem:.0f} MB ({max_mem/1024:.2f} GB, {mem_usage_pct:.2f}% of system)")
                    
                    if show_threads:
                        try:
                            threads = process.threads()
                            print(f"  [Thread Details] Total {len(threads)} threads:")
                            for i, thread in enumerate(threads[:10]):  # Show only first 10
                                print(f"    Thread {i+1}: ID={thread.id}")
                            if len(threads) > 10:
                                print(f"    ... and {len(threads) - 10} more threads")
                        except:
                            pass
                    
                    # Show detailed memory information
                    try:
                        mem_full = process.memory_full_info()
                        print(f"  [Memory Details] RSS={format_bytes(mem_full.rss)}, "
                              f"VMS={format_bytes(mem_full.vms)}, "
                              f"Shared={format_bytes(mem_full.shared)}")
                    except:
                        pass
                    
                    print()
                
        except KeyboardInterrupt:
            max_cores_used = max_cpu / 100.0
            cpu_usage_pct = (max_cpu / 100.0) / cpu_count * 100
            mem_usage_pct = (max_mem / 1024) / mem_total * 100
            print(f"\n\n{'='*90}")
            print(f"Monitoring stopped (user interrupt)")
            print(f"Final Statistics:")
            print(f"  Max CPU usage: {max_cpu:.1f}% ({max_cores_used:.1f} cores, {cpu_usage_pct:.2f}% of {cpu_count} CPUs)")
            print(f"  Max memory usage: {max_mem:.0f} MB ({max_mem/1024:.2f} GB, {mem_usage_pct:.2f}% of {mem_total:.1f} GB)")
            print(f"{'='*90}\n")
        except psutil.NoSuchProcess:
            print(f"\nProcess {pid} has ended")
        except Exception as e:
            print(f"\nError: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            
    except psutil.AccessDenied:
        print(f"Error: No permission to access process {pid} (may need root privileges)")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")


def list_matching_processes(script_name):
    """List all matching processes"""
    script_name = script_name.strip()
    if not script_name.endswith('.py'):
        script_name = script_name + '.py'
    
    # Get current process ID to exclude self
    current_pid = os.getpid()
    
    # Extract just the filename from path
    script_basename = os.path.basename(script_name)
    
    matching_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'username', 'create_time']):
        try:
            # Skip current process (monitor_retrieval.py itself)
            if proc.pid == current_pid:
                continue
            
            cmdline = proc.info['cmdline']
            if cmdline is None or len(cmdline) == 0:
                continue
            
            if 'python' not in proc.info['name'].lower():
                continue
            
            cmdline_str = ' '.join(cmdline)
            
            # Check if the script name appears in the command line
            if script_basename in cmdline_str:
                # Make sure it's not the monitor script itself
                if 'monitor_retrieval.py' in cmdline_str:
                    continue  # Skip monitor_retrieval.py itself
                
                matching_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    
    if len(matching_processes) == 0:
        print(f"No process found running '{script_name}'")
        return None
    else:
        print(f"Found {len(matching_processes)} matching process(es):\n")
        for i, proc in enumerate(matching_processes, 1):
            try:
                cmdline = ' '.join(proc.info['cmdline'][:3])
                create_time = time.strftime('%Y-%m-%d %H:%M:%S', 
                                           time.localtime(proc.info['create_time']))
                print(f"  {i}. PID={proc.pid}, User={proc.info['username']}, "
                      f"Start Time={create_time}")
                print(f"     Command: {cmdline}...")
            except:
                pass
        print()
        return matching_processes[0] if len(matching_processes) == 1 else None


def main():
    parser = argparse.ArgumentParser(
        description='Monitor resource usage of Python scripts (CPU, memory, thread count, etc.)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor specified Python script (script name only, path is optional)
  python monitor_retrieval.py crires_retrieval_chips_starB.py
  # or with path:
  python monitor_retrieval.py src/crires_retrieval_chips_starB.py
  
  # Specify update interval (seconds)
  python monitor_retrieval.py crires_retrieval_chips_starA.py --interval 5
  
  # Show thread details
  python monitor_retrieval.py crires_retrieval_chips_starB.py --show-threads
  
  # Directly specify process ID
  python monitor_retrieval.py --pid 12345
  
  # List all matching processes
  python monitor_retrieval.py crires_retrieval_chips_starB.py --list
        """
    )
    
    parser.add_argument(
        'script_name',
        nargs='?',
        help='Python script filename to monitor (e.g., crires_retrieval_chips_starB.py or src/crires_retrieval_chips_starB.py)'
    )
    
    parser.add_argument(
        '--pid',
        type=int,
        help='Directly specify process ID to monitor (if specified, script_name will be ignored)'
    )
    
    parser.add_argument(
        '--interval',
        type=float,
        default=2.0,
        help='Update interval in seconds (default: 2.0)'
    )
    
    parser.add_argument(
        '--show-threads',
        action='store_true',
        help='Show detailed thread information'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='Only list matching processes, do not monitor'
    )
    
    args = parser.parse_args()
    
    # Check if psutil is available
    try:
        import psutil
    except ImportError:
        print("Error: psutil library is required")
        print("Please run: pip install psutil")
        sys.exit(1)
    
    # If PID is specified, use it directly
    if args.pid:
        try:
            process = psutil.Process(args.pid)
            if args.list:
                print(f"Process PID={args.pid} information:")
                print(f"  Command: {' '.join(process.cmdline()[:5])}")
                print(f"  User: {process.username()}")
                print(f"  Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(process.create_time()))}")
            else:
                monitor_process(process, interval=args.interval, show_threads=args.show_threads)
        except psutil.NoSuchProcess:
            print(f"Error: Process {args.pid} does not exist")
            sys.exit(1)
        except psutil.AccessDenied:
            print(f"Error: No permission to access process {args.pid}")
            sys.exit(1)
        return
    
    # If script_name is not specified, show help
    if not args.script_name:
        parser.print_help()
        sys.exit(1)
    
    # Find process
    if args.list:
        list_matching_processes(args.script_name)
        return
    
    process = find_process_by_script_name(args.script_name)
    
    if process is None:
        print(f"Error: No process found running '{args.script_name}'")
        print(f"\nHint: Make sure the script is running, or use --list option to see all matching processes")
        sys.exit(1)
    
    # Start monitoring
    monitor_process(process, interval=args.interval, show_threads=args.show_threads)


if __name__ == '__main__':
    main()
