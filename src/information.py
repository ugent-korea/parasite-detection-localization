
def write_to_csv(file_name, arr):
    file_to_write = open(file_name, 'a')
    for item in arr:
        file_to_write.write(str(item)+',')
    file_to_write.write('\n')
    file_to_write.close()
    return 1