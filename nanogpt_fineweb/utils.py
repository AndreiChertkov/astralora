from datetime import datetime
import os
import shutil


def init_log(msg="", fpath="log.txt", enable=True):
    def log(text, kind="", only_file=False):
        if not enable:
            return

        pref = ""

        # if kind != 'ini' and kind != 'log':
        #    pref += '[' + datetime.now().strftime('%H-%M-%S') + '] > '

        if kind == "prc":
            pref = "... " + pref
        if kind == "res":
            pref = "+++ " + pref
        if kind == "wrn":
            pref = "WRN " + pref
        if kind == "err":
            pref = "!!! " + pref

        text = pref + text
        with open(fpath, "w" if kind == "ini" else "a+", encoding="utf-8") as f:
            f.write(text + "\n")
        if not only_file:
            print(text)

    dt = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    content = "Start computations"
    text = f"[{dt}] >> {content}"
    text += "\n" + "=" * 24 + " " + "-" * len(content) + "\n"
    if msg:
        text += msg
        text += "\n" + "=" * (25 + len(content)) + "\n"
    log(text, "ini")

    return log


def init_path(name, root='result', rewrite=False):
    os.makedirs(root, exist_ok=True)
    fold = f'{root}/{name}'
    if os.path.isdir(fold):
        if rewrite:
            act = 'y'
        else:
            msg = f'Path "{fold}" already exists. Remove? [y/n] '
            act = input(msg)
        if act == 'y':
            shutil.rmtree(fold)
        else:
            raise ValueError('Folder with results is already exists')
    os.makedirs(fold)