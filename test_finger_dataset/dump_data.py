import wget
import sys
import os

base_link = "https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100295/mat_data/"
# https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100295/mat_data/s08.mat

out_path = "/home/kubasinska/dataset/finger"
# todo: chnage path

def bar_progress(current, total, width=80):
  progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
  # Don't use print() as it will print in new line every time.
  sys.stdout.write("\r" + progress_message)
  sys.stdout.flush()

for i in range(1,53):
    if len(str(i)) == 1:
        name = "s0" + str(i) + ".mat"
    else:
        name = "s" + str(i) + ".mat"
    link = base_link + name
    print("\nDumping: " + name)


    new_file = wget.download(link, out=os.path.join(out_path, name), bar=bar_progress)
