import os


def dump_list_to_txt(l, file_name):
    if not os.path.exists('logs'):
        os.makedirs('logs')
    path = f'logs/{file_name}'

    with open(path, "a") as f:
        for i in l:
            f.write(f"{i}\n")

