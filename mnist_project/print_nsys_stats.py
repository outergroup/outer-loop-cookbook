import sqlite3
from collections import defaultdict

import numpy as np

def fetch_events_during_benchmark(db_path):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Fetch textId for "benchmark"
    cursor.execute("SELECT rowid FROM StringIds WHERE value = 'benchmark'")
    benchmark_text_id = cursor.fetchone()
    if benchmark_text_id is None:
        print("No 'benchmark' range found.")
        return
    benchmark_text_id = benchmark_text_id[0]

    # Fetch start and end times for "benchmark"
    cursor.execute(f"SELECT MIN(start), MAX(end) FROM NVTX_EVENTS WHERE textId = {benchmark_text_id}")
    start_time, end_time = cursor.fetchone()
    if start_time is None or end_time is None:
        print("No 'benchmark' range found.")
        return

    # Fetch textId for "cudaLaunchKernel"
    cursor.execute("SELECT rowid FROM StringIds WHERE value LIKE '%cudaLaunchKernel%'")
    cuda_launch_kernel_text_id = cursor.fetchone()
    if cuda_launch_kernel_text_id is None:
        print("No 'cudaLaunchKernel' events found.")
        return
    cuda_launch_kernel_text_id = cuda_launch_kernel_text_id[0]

    # Fetch textId for "cudaDeviceSynchronize"
    cursor.execute("SELECT rowid FROM StringIds WHERE value LIKE '%cudaDeviceSynchronize%'")
    cuda_device_synchronize_text_id = cursor.fetchone()
    if cuda_device_synchronize_text_id is None:
        print("No 'cudaDeviceSynchronize' events found.")
        return
    cuda_device_synchronize_text_id = cuda_device_synchronize_text_id[0]

    # Fetch all cudaLaunchKernel events within the "benchmark" time range
    cursor.execute(f"SELECT start, end FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE nameId = {cuda_launch_kernel_text_id} AND start >= {start_time} AND end <= {end_time}")
    cuda_launch_kernel_events = cursor.fetchall()

    # Fetch all cudaDeviceSynchronize events within the "benchmark" time range
    cursor.execute(f"SELECT start, end FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE nameId = {cuda_device_synchronize_text_id} AND start >= {start_time} AND end <= {end_time}")
    cuda_device_synchronize_events = cursor.fetchall()

    all_events = defaultdict(list)
    unique_names = defaultdict(set)

    table = 'CUPTI_ACTIVITY_KIND_KERNEL'
    # Fetch all events within the "benchmark" time range
    cursor.execute(f"SELECT start, end, shortName FROM {table} WHERE start >= {start_time} AND end <= {end_time}")
    events = cursor.fetchall()

    for start, end, name_id in events:
        # Append to the list of events
        all_events[table].append((start, end))
        unique_names[table].add(name_id)

    # Other tables to fetch events from
    tables = [
        'CUPTI_ACTIVITY_KIND_MEMCPY',
    ]

    for table in tables:
        # Fetch all events within the "benchmark" time range
        cursor.execute(f"SELECT start, end FROM {table} WHERE start >= {start_time} AND end <= {end_time}")
        events = cursor.fetchall()

        for start, end in events:
            # Append to the list of events
            all_events[table].append((start, end))

    all_unique_names = set()
    for table, table_unique_names in unique_names.items():
        for name_id in table_unique_names:
            # Fetch the name for this event
            cursor.execute(f"SELECT value FROM StringIds WHERE rowid = {name_id}")
            name = cursor.fetchone()
            all_unique_names.add(name[0])

    # Close the database connection
    conn.close()

    return start_time, end_time, cuda_launch_kernel_events, cuda_device_synchronize_events, all_events, all_unique_names


def durations(events):
    events = np.array(events)
    return events[..., 1] - events[..., 0]


def main(db_path):
     (start_time,
      end_time,
      cuda_launch_kernel_events,
      cuda_device_synchronize_events,
      device_events,
      unique_names) = fetch_events_during_benchmark(db_path)

     # We assume the first few launch kernel events didn't block.
     launch_kernel_active_time = durations(cuda_launch_kernel_events[10:40]).mean()
     total_lk_active_time = int(launch_kernel_active_time
                                * len(cuda_launch_kernel_events))
     estimated_launch_kernel_wait_time = max(
         0,
         durations(cuda_launch_kernel_events).sum() - total_lk_active_time)

     benchmark_duration = end_time - start_time
     estimated_cpu_wait_time = (durations(cuda_device_synchronize_events).sum()
                                + estimated_launch_kernel_wait_time)
     cpu_wait_pct = 100. * (1 - (estimated_cpu_wait_time / benchmark_duration))

     estimated_gpu_active_time = np.sum([durations(events).sum()
                                         for events in list(device_events.values())])
     gpu_wait_pct = 100. * estimated_gpu_active_time / benchmark_duration

     print(f'Duration: {benchmark_duration}')
     print(f'Estimated CPU wait time: {estimated_cpu_wait_time}')
     print(f'Estimated GPU active time: {estimated_gpu_active_time}')
     print(f'Estimated CPU usage: {cpu_wait_pct:>.2f}%')
     print(f'Estimated GPU usage: {gpu_wait_pct:>.2f}%')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('db_path', help='Path to the database file.')
    main(parser.parse_args().db_path)
