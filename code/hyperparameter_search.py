import os
import numpy as np
import foundations

NUM_JOBS = 10

def generate_params():

    hyper_params = {'batch_size': int(np.random.choice([8, 16, 32, 64])),
                    'epochs': int(np.random.choice([100, 200])),
                    'learning_rate': np.random.choice([0.1, 0.01, 0.001, 0.0001]),
                    'decoder_neurons': [np.random.randint(16, 512), np.random.randint(16, 512),
                                        np.random.randint(16, 512), np.random.randint(16, 512)],
                    }
    return hyper_params


for job_ in range(NUM_JOBS):
    print(f"packaging job {job_}")
    hyper_params = generate_params()
    foundations.submit(scheduler_config='scheduler', job_directory='.', command='main.py', params=hyper_params,
                       stream_job_logs=False)