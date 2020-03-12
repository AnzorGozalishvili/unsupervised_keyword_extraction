#!/bin/sh
bert-serving-start -num_worker=2 -max_seq_len=None -model_dir ./cased_L-12_H-768_A-12
