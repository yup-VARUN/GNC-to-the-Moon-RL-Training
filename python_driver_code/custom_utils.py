import inspect
from functools import partial
import pandas as pd
import json
import os


def prepare_agent(agent: object, keyargs:dict)-> partial:
    '''
    Prepares agent for multiprocess process to avoid passing direct args/class to child process
    agent := object where agent can run episodes, collect experiences with provided env
    keyargs := key arguments for 
    '''
    signature = inspect.signature(agent.__init__)
    bounded_args = {}

    # Process provided user args
    for arg in keyargs:
        if arg in signature.parameters:
            bounded_args[arg] = keyargs[arg]

    # test user provided args to if missing args
    try:
        signature.bind_partial(**bounded_args)
    except Exception as e:
        print("Unexpected binding error:",e)
        # print("Unknown User provided arg for agent:", arg)
        raise
    

    target_agent = partial(agent, **bounded_args)
    # for arg, value in signature.parameters.items():

    #     if arg in keyargs:
    #         target_agent = partial(target_agent, keyargs[arg])
    #     if isinstance(value, inspect.Parameter) and arg not in keyargs and arg != 'self':
    #         print("Missing necessary parameter in user provided keyargs!", arg)
    #         raise
        
    return target_agent

def log_dump_to_json(filename: str, log: dict):
    '''
    log schema:
            total_execution_time : float
            initial_mem_usage : in MB
            episodes_per_epoch : int
            Epoches:
                Epoch:
                    total_compute_time_for_episodes_cnt/spent_in_curr_epoch: float
                    total_num_SAR_collected: int
                    time_spent_waiting_on_update: float
                    
                    episodes:
                        [Episode:
                                Memory_usage_at_episode
                                
                            ...
                        Episode_n]
                ...
                Epoch:
                    ...
    '''
    with open(filename + ".json",'w') as file:
        json.dump(log, file)
    return f"{filename} has been dumped"

    pass

def view_worker_metrics(worker_json: str):
    with open(worker_json, "r") as file:
        log = json.load(file)

    episodes_per_epoch = log["episodes_per_epoch"]
    total_exec_time = log["total_execution_time"]
    initial_mem = log["init_mem_usage"]
    episodes_per_epoch = log["episodes_per_epoch"]

    # Aggregate over epochs
    total_compute_time = 0.0
    total_wait_time = 0.0
    total_episodes = 0

    epoch_summary = []
    for epoch_idx, epoch_data in log["Epoches"].items():
        episodes = epoch_data.get("Episodes", {})
        
        total_memory_usage = sum(episodes.values()) / episodes_per_epoch
        
        epoch_summary.append({
            "epoch": int(epoch_idx),  # JSON keys might be strings
            "total_compute_time": epoch_data["total_compute_time_for_episodes_cnt"],
            "SAR_collected": epoch_data["total_num_SAR_collected"],
            "time_waiting_on_update": epoch_data["time_spent_waiting_on_update"],
            "total_memory_usage": total_memory_usage
        })

    epoch_df = pd.DataFrame(epoch_summary)
    
    for epoch_data in log["Epoches"].values():
        total_compute_time += epoch_data["total_compute_time_for_episodes_cnt"]
        total_wait_time += epoch_data["time_spent_waiting_on_update"]
        total_episodes += len(epoch_data["Episodes"])

    # Avoid division by zero
    if total_episodes == 0:
        avg_compute_per_episode = 0.0
        avg_wait_per_episode = 0.0
    else:
        avg_compute_per_episode = total_compute_time / total_episodes
        avg_wait_per_episode = total_wait_time / total_episodes
    # Construct the summary as a DataFrame (1-row)
    summary_df = pd.DataFrame([{
        "total_execution_time": total_exec_time,
        "initial_mem_usage_MB": initial_mem,
        "episodes_per_epoch": episodes_per_epoch,
        "total_episodes": total_episodes,
        "avg_compute_time_per_episode": avg_compute_per_episode,
        "avg_wait_time_per_episode": avg_wait_per_episode,
    }])
    return epoch_df, summary_df
    pass

def view_all_workers(log_dir):
    summary_dfs = []
    epoch_dfs = []


    for root, _, files in os.walk(log_dir):
        for file in files:
            if file.endswith(".json"):
                full_path = os.path.join(root, file)
                # print(full_path)
                try:
                    epoch_df, summary_df = view_worker_metrics(full_path)
                    # print(summary_df)
                    # Add a file identifier for traceability
                    run_id = os.path.splitext(file)[0]
                    summary_df["run_id"] = run_id
                    epoch_df["run_id"] = run_id

                    summary_dfs.append(summary_df)
                    epoch_dfs.append(epoch_df)

                except Exception as e:
                    print(f"Error processing {full_path}: {e}")
    print(summary_dfs)
    all_summaries = pd.concat(summary_dfs, ignore_index=True)
    all_epochs = pd.concat(epoch_dfs, ignore_index=True)

    return all_epochs, all_summaries

def summarize_across_workers(summary_df):

    agg = {
        "total_execution_time": ["mean", "std", "min", "max"],
        "initial_mem_usage_MB": ["mean", "std"],
        "total_episodes": ["mean", "sum"],
        "avg_compute_time_per_episode": ["mean", "std"],
        "avg_wait_time_per_episode": ["mean", "std"]
    }

    summary_stats = summary_df.agg(agg)

    # # Flatten MultiIndex columns: ('col', 'agg') -> 'col_agg'
    # # If result has MultiIndex columns, flatten them
    # if isinstance(summary_stats.columns, pd.MultiIndex):
    #     summary_stats.columns = [
    #         f"{col}_{stat}" for col, stat in summary_stats.columns
    #     ]
    
    # # Reset to get a single-row dataframe
    # summary_stats = summary_stats.reset_index(drop=True)

    # Swap axes: now rows = aggregation functions, columns = metric names
    # Swap rows and columns so aggfunc becomes the index
    summary_stats = summary_stats.T.unstack().unstack()

    # Now 'summary_stats' has index = aggfuncs, columns = metrics
    summary_stats.index.name = "aggregate_function"
    summary_stats.columns.name = None  # for clean display
    return summary_stats
    
import matplotlib.pyplot as plt

def plot_worker_metric_over_epochs(epoch_df, worker_name, metric):
    # Check if metric is valid
    if metric not in epoch_df.columns:
        raise ValueError(f"Metric '{metric}' not found in DataFrame columns.")

    # Filter for the specific worker
    worker_df = epoch_df[epoch_df["run_id"] == worker_name]

    if worker_df.empty:
        raise ValueError(f"No data found for worker '{worker_name}'.")

    # Ensure sorted by epoch
    worker_df = worker_df.sort_values("epoch")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(worker_df["epoch"], worker_df[metric], marker='o')
    plt.title(f"{metric.replace('_', ' ').title()} Over Epochs - {worker_name}")
    plt.xlabel("Epoch")
    plt.ylabel(metric.replace("_", " ").title())
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_metrics_from_single_log(epoch_df, metrics):
    if isinstance(metrics, str):
        metrics = [metrics]

    plt.figure(figsize=(12, 6))

    for metric in metrics:
        if metric not in epoch_df.columns:
            print(f"⚠️ Metric '{metric}' not found in epoch_df. Skipping.")
            continue
        plt.plot(epoch_df["epoch"], epoch_df[metric], marker="o", label=metric.replace("_", " ").title())

    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title("Worker Metrics Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()