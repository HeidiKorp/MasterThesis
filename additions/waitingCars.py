counter = 0
c = 0
prev_obj_id = -2

fileName = "../../intersections-dataset.csv"
# fileName = "../../records/records_0-5000.csv"

out = open("waiting-dataset.csv", "a")

with open(fileName) as infile:
    for line in infile:
        if c == 0:
            out.write(line)
            print("First line? ", line)
        s = line.split(",")
        if s[3] != prev_obj_id and (s[24] == "0.0"
                                    or s[25] == "0.0"
                                    or s[28] == "0.0"
                                    or s[29] == "0.0"):
            out.write(line)
            counter += 1
            prev_obj_id = s[3]
        c += 1
print("Tracks: ", counter)
out.close()