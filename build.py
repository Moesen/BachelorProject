import os

# Requires cmake, mingw and make
os.chdir("./GEL/build/")
if len(os.listdir()) > 2:
    print("Wow")