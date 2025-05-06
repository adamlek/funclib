import os
from typing import (
    Callable, 
    Any, 
    Tuple, 
    Iterable, 
    )
from collections import defaultdict
import json
import itertools
import functools
import operator
import random


import toolz # type: ignore[import]
from IPython import embed

# custom type signatures
StrListToStr = Callable[[list[str], int, str], list[str]]

def f_append(list, item):
    """
    do list += item as a function
    """
    return list.__iadd__(item)

def insert_at_positions(input: list[str], 
                        num_inserts: int, 
                        insert_text: str
                        ) -> list[str]:
    """
    Insert `insert_text` after the token at position 
    `input[i]` do this `num_inserts` times  
    """
    after_ids = random.sample(range(len(input)), k=num_inserts)
    
    return list(
        toolz.concat(
            [(x, insert_text) if i in after_ids # a bit fuck-y with nested list
              else (x) 
              for i, x in enumerate(input)]))
    
def replace_at_positions(input: list[str], 
                         num_replacements: int, 
                         insert_text: str
                         ) -> list[str]:
    """
    Replace `input[i]` with `insert_text`, do this `num_replacements` times
    """
    after_ids = random.sample(range(len(input)), k=num_replacements)
    return [insert_text if i in after_ids 
            else x 
            for i, x in enumerate(input)]
    
def mask_text_with_func(text,
                        mask_func: StrListToStr, 
                        num_masks: int = 1, 
                        mask_token: str = '<m>',
                        tokenize: bool | Callable = False
                        ) -> Any:
    """
    Take a text, tokenize if such a function is provided, 
    and apply `mask_func` to the sequence
    """
    if tokenize:
        text = tokenize(text) # type: ignore
    return mask_func(text, num_masks, mask_token)

def group_list_of_dicts_by_key(iterable: list[dict[Any, Any]], 
                               key_func: Callable
                               ) -> dict[Any, list[dict[Any, Any]]]:
    """
    raise the values of a key. 
        {a:1, b:0}, ..., {a:0, b:1} (values of a into keys)-> {1: {...}, 2: {...}}
    """
    grouped_data = {}
    sorted_iterable = sorted(iterable, key=key_func)

    for key, group in toolz.groupby(key=key_func, seq=sorted_iterable).items():
        grouped_data[key] = list(group)
        
    return grouped_data

def balance_dataset(dataset: list[Any],
                    label_key: str,
                    sample_func: Callable,
                    ) -> itertools.chain[Any]:
    """
    Balance a dataset according to the class with fewest examples
    
    Arguments:
        dataset: list (of dicts...)
        label_key: the key of the classes/labels in the dataset
        sample_func: the function which samples from the dataset, 
                     ensure that it has a k-argument (for how many
                     samples to take).
                     
    Output:
        An unshuffled iterator containing the balanced dataset, 
        you should shuffle the dataset before training with it
    """
    
    partitioned_dataset: dict[Any, list[Any]] = \
        group_list_of_dicts_by_key(
            dataset, 
            operator.itemgetter(label_key)
            )

    label_distribution = toolz.valmap(len, partitioned_dataset)
    _, num_samples = sorted(label_distribution.items(), key=lambda x: x[1])[0]
    
    data = toolz.valmap(functools.partial(sample_func, k=num_samples), 
                        partitioned_dataset)
    
    return itertools.chain.from_iterable(data.values())


def format_dict_batch(batch: list[Any], 
                      key_order: list[str],
                      key_funcs: dict[str, Tuple[Tuple, Any]]
                      ) -> dict[str, list[Any]]:
    """
    Arguments:
        batch: list of dicts
        key_order: order of the keys in individual batches
        key_funcs: dict[str -> name in output dict,
                        Tuple[Any -> name of key in individual dicts,
                              Any -> function to apply to values in dicts]
                       ]
    Output:
        Dict containing {key_func.keys(): list(map(lambda x: func(x), values)), ...}
    """
    
    output: dict[str, list] = defaultdict(list)
    vs: list[Any] = list(zip(*[x.values() for x in batch]))
    ks_vs = {k: vs[i] for i, k in enumerate(key_order)}
    
    for key, process in key_funcs.items():
        apply_to_keys, func = process
        if len(apply_to_keys) == 1: # applying function to values from ONE key
            output[key] = list(map(lambda x: func(x), 
                                   ks_vs[toolz.first(apply_to_keys)]))
        else: # applying function to values from many keys
            output[key] = [func(*vals) for vals in 
                           zip(*toolz.keyfilter(lambda x: x in apply_to_keys, 
                                                ks_vs).values()
                               )
                           ]
    return output

def get_sampled_batches_from_dataset(data: list,
                                     batch_size: int, 
                                     sample_func: Callable,
                                     batch_filter_func: Callable = lambda x: x,
                                     do_shuffle: bool = True,
                                     batch_filter_func_kwargs: dict = {},
                                     sample_func_kwargs: dict = {}
                                    ):
    if do_shuffle:
        random.shuffle(data)
    
    num_steps = int(len(data)/batch_size)+1
    for _ in range(num_steps):
        batch = sample_func(data, **sample_func_kwargs)
        if batch:
            yield batch_filter_func(batch, **batch_filter_func_kwargs)
    
def get_linear_batches_from_dataset(data: list, 
                                    batch_size: int, 
                                    batch_filter_func: Callable = lambda x: x,
                                    do_shuffle: bool = True,
                                    batch_filter_func_kwargs: dict = {},
                                    ) -> Iterable:
    """
    Get batches linearly from a list of batches [dict_0, ..., dict_n]
    
    Arguments:
        data: dataset(list) of items (list, dict, tuple, ...)
        batch_size: how many items in each batch?
        batch_filter_func: function to process the items in each batch 
        do_shuffle: whether or not to shuffle the dataset before generating batches
        
    Output:
        a batch of items that have been processed by `batch_filter_func` 
    """
    
    if do_shuffle:
        random.shuffle(data)
    
    num_steps = int(len(data)/batch_size)+1
    iter_data = iter(data) # ignore_type
    
    for _ in range(num_steps):
        batch = list(toolz.take(batch_size, iter_data))
        if batch:
            yield batch_filter_func(batch, **batch_filter_func_kwargs)

def read_files_in_folder(folder: str, 
                         file_read_func: Callable, 
                         file_name_func: Callable = lambda x: x
                         ) -> Iterable[Tuple[str, Any]]:
    """
    Arguments:
        folder: the folder to read
        file_read_func: how to read the file
        file_name_func: how to process the filename
        
    Output:
        tuple(processed_filename, file_contents)_0, ...
    """
    for file in os.listdir(folder):
        with open(folder+file) as f:
            yield (file_name_func(file), file_read_func(f))
            
def write_jsonl_file(file_path: str, 
                     file_flags: str,
                     data: list, 
                     data_process_func: Callable = lambda x: x, 
                     **json_kwargs
                     ) -> bool:
    with open(file_path, file_flags) as out:
        for datapoint in data:
            out.write(json.dumps(data_process_func(datapoint), 
                                 **json_kwargs))
    return True