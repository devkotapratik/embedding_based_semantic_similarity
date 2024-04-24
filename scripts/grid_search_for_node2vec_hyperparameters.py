import subprocess
import multiprocessing
import time
from pathlib import Path
from itertools import product
from tqdm import tqdm


max_processes = 10 # Maximum processes allowed during multiprocessing
delay = 10 # in minutes - time to wait (if max. processes running concurrently) before checking for availability
            # wait for 'delay' minutes before checking if a slot is available for parallel processing

def train_embeddings(args_queue, result_queue, proc_count):
    # Run NODE2VEC_on_UBERON.py script using subprocess to run multiple instances
    # each instance will have different set of hyperparameters pulled from 'args_queue'
    # The result of the process is pushed to 'result_queue'. 
    try:
        started = False # variable to check if the process started (will not start if max_processes reached)
        semaphore.acquire() # enter a critical section and decrease semaphore internal counter
        args = args_queue.get() # get hyperparameter from 'args_queue'
        log_file = Path.joinpath(
            Path(".").absolute(), "grid_search", "_".join([args[1], args[3], args[5], args[7]]) + ".log")
        with proc_count.get_lock():
            if proc_count.value < max_processes: # Call the node2vec script only if num. of running processes < max.
                proc_count.value += 1
                with open(log_file, "w") as log:
                    command = [
                        "python", "NODE2VEC_on_UBERON.py"] + args
                    # print(command)
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    started = True
        if started:
            process.wait() # If process started, wait for it to finish before returning back to calling method
            if process.returncode != 0: # Some error occured, pass the error return code
                result_queue.put((False, args, f"Process {process.pid} exited with non-zero exit code: {process.returncode}"))
                args_queue.put(args) # if error occured, put the arguments back into the queue
            else:
                result_queue.put((True, args, f"Process {process.pid} executed successfully"))
    except Exception as ex:
        result_queue.put((False, args, str(ex)))
        args_queue.put(args) # if exception, put the arguments back into the queue
    finally:
        semaphore.release() # Exit the critical section, increase interal counter
        with proc_count.get_lock():
            proc_count.value -= 1 # Decrease the num. of running processes


if __name__ == "__main__":
    BASE_DIR = Path(".").absolute().parent
    DATA_DIR = Path.joinpath(BASE_DIR, "data")
    EMB_DIR = Path.joinpath(DATA_DIR, "grid_search", "embeddings")
    EMB_DIR.mkdir(exist_ok=True, parents=True)
    
    # walk_number = list(range(50, 160, 10))
    # walk_length = list(range(5, 35, 5))
    # p = [round(i*0.1, 1) for i in range(10, 21)]
    # q = [round(i*0.1, 1) for i in range(1, 10)]

    # Search space for hyperparameter optimization for NODE2VEC embeddings
    # 600 possible combinations
    walk_number = list(range(50, 150, 30))
    walk_length = [5, 10, 20, 30, 35]
    p = [round(i*0.1, 1) for i in range(10, 21, 2)]
    q = [round(i*0.1, 1) for i in range(1, 10, 2)]

    arg_space = [
        [
            "--walk_number", str(i[0]), "--walk_length", str(i[1]),
            "--p", str(i[2]), "--q", str(i[3])
        ] for i in product(walk_number, walk_length, p, q)
    ]

    # Push arguments (to be passed to the NODE2VEC script) into a multiprocessing queue
    args_queue = multiprocessing.Queue()
    for args in arg_space:
        args_queue.put(args)
    result_queue = multiprocessing.Queue() # To put results from each process
    processes = []

    semaphore = multiprocessing.Semaphore(max_processes) # number of max processes
    proc_count = multiprocessing.Value("i", 0) # count for current processes
    successes, failures = [], []
    base_desc = "Running/Total processes"
    pbar = tqdm(total=len(arg_space), desc=base_desc)
    while True:
        with proc_count.get_lock():
            running = proc_count.value
        if running >= max_processes: # Wait for 'delay' minutes before checking again if currently running processes
            # are less than max. allowed parallel processes. A NODE2VEC algorithm typically for our configuration takes
            # between 10 to 30 minutes, so put delay accordingly so that there is no need to constantly check if a new 
            # process can start.
            for i in range(60*delay, -1, -1):
                new_desc = base_desc + f" | Max. allowed processes reached | waiting for {i} seconds"
                pbar.set_description(new_desc)
                time.sleep(1)
        else:
            if args_queue.qsize() == 0: # No more arguments in the 'args_queue' meaning all possible combinations are used
                                        # to train embedding model
                # 'Args_queue' might be empty but some process might still be running, wait for running process to finish
                while True: # Check for proc_count to see if all processes finished execution
                    time.sleep(1)
                    prev_running = running
                    with proc_count.get_lock():
                        running = proc_count.value
                    if running == 0: # Exit if no more process running
                        break
                    else:
                        if prev_running != running:
                            pbar.update(prev_running - running)
            else: # 'args_queue' not empty and max_process not reached -> spawn a new process
                process = multiprocessing.Process(target=train_embeddings, args=(args_queue, result_queue, proc_count))
                process.start()
                processes.append(process)
                pbar.set_description(base_desc)
                n_finished = len([i for i in processes if not i.is_alive()])
                pbar.update(1)
                time.sleep(1)
        with proc_count.get_lock():
            running = proc_count.value
            if args_queue.qsize() == 0 and running == 0: # Exit when all possible hyperparameter settings explored
                break
    
    pbar.set_description("Finishing all processes")
    while len(processes): # To make sure than 
        for process in processes:
            if process.is_alive(): # If some process is still somehow alive, force kill the process
                process.terminate()
            processes.remove(process) # Remove process from the list
            break
        time.sleep(1)
    
    # To check how many spawned processes ran successfully and how many failed, can have failed+successful attempts >
    # total possible combinations of hyperparameters as it is possible to have multiple failed attempts for the same
    # set of hyperparameters before finally succedding. 
    for _ in range(len(arg_space)):
        success, args, result = result_queue.get()
        if success:
            successes.append((args, result))
        else:
            failures.append((args, result))
    print(f"\n\nTotal failed attempts: {len(failures)}\nTotal successful attempts: {len(successes)}")
