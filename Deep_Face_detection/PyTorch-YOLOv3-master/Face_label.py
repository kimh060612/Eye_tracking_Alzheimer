import os

txt_path = "/home/michael/FDDB/FDDB-folds/"
txtList = os.listdir(txt_path)

topDir = "/home/michael/FDDB"
total_annotation = []

for con in txtList:
    with open(txt_path+con, "r") as File:
        Path = File.readlines()
        for contents in Path:
            tmp = contents.split("/")
            re = "/".join(tmp)
            re = topDir + re
            total_annotation.append(re)

outPath = "/home/michael/FDDB/FDDB-folds/FDDB_paths.txt"
with open(outPath, "w") as File:
    File.writelines(total_annotation)

