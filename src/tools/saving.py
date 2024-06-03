import os

def base_path_creator(core_args, create=True):
    path = '.'
    path = next_dir(path, 'experiments', create=create)
    path = next_dir(path, core_args.data_name, create=create)
    path = next_dir(path, core_args.model_name, create=create)
    path = next_dir(path, core_args.task, create=create)
    path = next_dir(path, core_args.language, create=create)
    path = next_dir(path, f'seed{core_args.seed}', create=create)
    return path

def create_safety_base_path(safety_args, path='.', mode='train', create=True):
    # Choose the base directory based on mode
    base_dir = 'safety_train' if mode == 'train' else 'safety_eval'
    path = next_dir(path, base_dir, create=create)

    # Common directory structure for both train and eval
    path = next_dir(path, safety_args.safety_method, create=create)
    path = next_dir(path, f'prepend_size{int(safety_args.prepend_size)}', create=create)
    path = next_dir(path, f'clip_val{float(safety_args.clip_val)}', create=create)

    # Additional directory for eval mode
    if mode == 'eval':
        path = next_dir(path, f'safety-epoch{int(safety_args.safety_epoch)}', create=create)
    
    return path

def safety_base_path_creator_train(safety_args, path='.', create=True):
    return create_safety_base_path(safety_args, path, 'train', create)

def safety_base_path_creator_eval(safety_args, path='.', create=True):
    return create_safety_base_path(safety_args, path, 'eval', create)

def next_dir(path, dir_name, create=True):
    if not os.path.isdir(f'{path}/{dir_name}'):
        try:
            if create:
                os.mkdir(f'{path}/{dir_name}')
            else:
                raise ValueError("provided args do not give a valid model path")
        except:
            # path has already been created in parallel
            pass
    path += f'/{dir_name}'
    return path
