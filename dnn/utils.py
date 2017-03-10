import os
import tensorflow as tf
import time


def clean_output(directory, prefix, max_to_keep, interval):
    while True:
        time.sleep(max(interval, 1))
    
        # Get candidates
        filtered = []
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if filename.startswith(prefix):
                    filtered.append(os.path.join(root, filename))
        print("Filterd files:", filtered)

        # Remove the oldest
        length = len(filtered)
        if length > max_to_keep:
            filtered.sort(key = os.path.getmtime)
            for i in range(length - max_to_keep):
                print("Deleting", filtered[i])
                os.remove(filtered[i])


def latest_checkpoint(checkpoint_dir, latest_filename="checkpoint"):
    checkpoint_path = os.path.join(checkpoint_dir, latest_filename)
    with open(checkpoint_path, "r") as fin:
        lines = fin.readlines()
    
    # Rewrite the checkpoint file to avoid wrong path
    absolute_path = lines[0].split("\"")[1]
    file_name = absolute_path.split("/")[-1].split("\\")[-1] # Cross-platform issue
    with open(checkpoint_path, "w") as fout:
        fout.write("model_checkpoint_path: \"%s\"\n" % file_name)
        fout.write("all_model_checkpoint_paths: \"%s\"\n" % file_name)
    
    return tf.train.latest_checkpoint(checkpoint_dir)


def real_type(FLAGS):
    return tf.float16 if FLAGS.use_fp16 else tf.float32


def print_model():
    print("Local variables are:")
    for v in tf.local_variables():
        print("parameter:", v.name, "device:", v.device, "shape:", v.get_shape())
    print("------------------------------------------------")
    
    print("Global variables are:")
    for v in tf.global_variables():
        print("parameter:", v.name, "device:", v.device, "shape:", v.get_shape())
    print("------------------------------------------------")
    
    print("Trainable variables are:")
    for v in tf.trainable_variables():
        print("parameter:", v.name, "device:", v.device, "shape:", v.get_shape())
    print("------------------------------------------------")
