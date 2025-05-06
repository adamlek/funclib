from data import (
    format_dict_batch, 
    get_linear_batches_from_dataset,
    balance_dataset,
    mask_text_with_func,
    insert_at_positions,
    replace_at_positions
)

import operator
import toolz # type: ignore[import]

import string
import random
from typing import (
    Any,
    Tuple,
    Callable
)
from IPython import embed

if __name__ == "__main__":
    
    
    # Create a simple dataset of dicts to play with
    data : list[dict[str, Any]] = [
        {'text': ''.join(random.choices(string.ascii_lowercase + string.digits, k=10)), 
         'label': random.randint(0,1)} 
        for _ in range(20)
        ]
    
    # Define how each key in the dataset of dicts should be processed
    # The keys in kfs will define the structure of the output dictionary
    # The values of each key in the output dictionary will be:
    # Apply the function (kfs[key][1]) to the values of 
    # keys (kfs[0]) in the list of dictionaries
    kfs: dict[str, Tuple[Tuple, Callable]] = {
        'text': (tuple(['text']), lambda x: x.upper()),
        'labels': (tuple(['label']), lambda x: float(x)),
        'label_and_text': (tuple(['text', 'label']), lambda x, y: x+'_'+str(y))
        }
    
    # Obtain batches from the dataset by taking n items and yielding them
    # Each batch is processed the the function `format_dict_batch` which have
    # the arguments from `batch_filter_func_kwargs` applied to it
    
    # Alternatively, you can partially apply the arguments:  
    # partial_format_dict_batch = partial(format_dict_batch, 
                                        # key_order=['text', 'label'],
                                        # key_funcs=kfs)
    for x in get_linear_batches_from_dataset(
        data,
        batch_size=5,
        batch_filter_func=format_dict_batch,
        batch_filter_func_kwargs={'key_order':['text', 'label'], 'key_funcs':kfs},
        do_shuffle=True,
        ):
        
        for item in x['text']:
            # Add a mask *after* random positions in the string
            masked = mask_text_with_func(
                item, 
                mask_func=insert_at_positions, 
                num_masks=10, 
                mask_token='<m>',
                tokenize=lambda x: list(x)
                )
            print('Inserting at random positions:', item, masked)
            
            #Replace random positions in the string with mask
            masked = mask_text_with_func(
                item, 
                mask_func=replace_at_positions, 
                num_masks=9, 
                mask_token='<m>',
                tokenize=lambda x: list(x)
                )
            print('Replacing at random positions:', item, masked)
        
    # You can balance the dataset with the following function
    # It will find the class with the fewest instances and 
    # sample that many examples from the dominant classes
    # b = balance_dataset(data,
    #                     label_key='label',
    #                     sample_func=random.choices,
    #                     )

