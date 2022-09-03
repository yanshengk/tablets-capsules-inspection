import os


def print_message(message, category="    "):
    os.system("date")
    print(f"\033[1m[{category}]\033[0m  {message}")


def make_directory(parent, child):
    path = os.path.join(parent, child)

    if os.path.exists(path):
        print_message(f"\"{child}\" directory already exists", "INFO")
    else:
        os.mkdir(path)
        print_message(f"Successfully created directory \"{child}\"", "INFO")
