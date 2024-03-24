import numpy as np
import os
from collections import Counter
import re
from datetime import datetime

def list_files(path):

    all_files = list()
    for (dirpath,_,filenames) in os.walk(path):
        all_files += [os.path.join(dirpath,file) for file in filenames if file.endswith('.logtxt')]
    return all_files


def read_file(file_path, word_counts, warning_timestamps):

    N_nonempty_line = 0
    N_error_files = 0

    with open(file_path, 'r') as file:
        error_found = False

        for line in file:

            stripped_line = line.strip()
            if stripped_line:
                N_nonempty_line += 1
                
                if '---' in line:
                    ind = line.find('---')
                    entry_info = stripped_line[:ind]
                    entry_msg = stripped_line[ind+3:]
                else:
                    indicies = [m.end() - 1 for m in re.finditer(r'[^a-zA-Z\s\-;]', line)]
                    ind = max(indicies) if indicies else -1
                    entry_info = stripped_line[:ind+1]
                    entry_msg = stripped_line[ind+1:]

                if (not error_found) and ("err" in entry_info.lower()):
                    N_error_files += 1
                    error_found = True

                if ind != -1:
                    if ';' in entry_msg:
                        entry_msg = entry_msg.replace(';','')
                    
                    words = list(set(entry_msg.split()))
                    if '-' in words:
                        words = [word for word in words if word != '-']
                    word_counts.update(words)

                if 'warn' in entry_info.lower():
                    if entry_info.startswith('['):
                        time_format = r'[%Y-%m-%d %H:%M:%S]'
                        timestamp = datetime.strptime(entry_info[:21], time_format)
                    elif entry_info.startswith('dt='):
                        entry_info = entry_info.replace('dt=','')
                        time_format = r'%Y-%m-%d_%H:%M:%S'
                        timestamp = datetime.strptime(entry_info[:19],time_format)
                    elif (entry_info[2] == '.'):
                        if (entry_info[13] != 'h'):
                            time_format = r'%d.%m.%Y.%H:%M:%S'
                            timestamp = datetime.strptime(entry_info[:19], time_format)
                        else:
                            time_format = r'%d.%m.%Y.%Hh:%Mm:%Ss'
                            timestamp = datetime.strptime(entry_info[:22], time_format)

                    elif (entry_info[4] == ' '):
                        time_format = r'%Y %m %d %H:%M:%S'
                        timestamp = datetime.strptime(entry_info[:19], time_format)

                    warning_timestamps.append(timestamp)

    return (N_nonempty_line, N_error_files)


def find_most_frequent_words(word_counts):

    word_counts = sorted(word_counts.items(), key=lambda x: x[0])
    word_counts.sort(key=lambda x: x[1], reverse=True)

    if len(word_counts) >= 5:
        top_words = word_counts[:5]
        top_words = [word_tuple[0] for word_tuple in top_words]
    else:
        top_words = word_counts[:]
        top_words = [word_tuple[0] for word_tuple in top_words]

    return top_words


def find_longest_period(warning_timestamps):
    warning_timestamps.sort()
    warning_timestamp_diffs = np.diff(warning_timestamps).astype('timedelta64[s]')
    if len(warning_timestamp_diffs) <= 4:
        t = np.sum(warning_timestamp_diffs)
        t = t.astype(int)
    else:
        t = 0
        for i in range(len(warning_timestamp_diffs)-4+1):
            t_tmp = np.sum(warning_timestamp_diffs[i:i+4])
            t_tmp = t_tmp.astype(int)
            if t_tmp > t:
                t = t_tmp
    return t


# ---main---

path = r'public1\set\10'
#path = input()
file_paths = list_files(path)

N_nonempty_lines = 0
N_error_files = 0
word_counts = Counter()
warning_timestamps = []

for file_path in file_paths:
    (N1, N2) = read_file(file_path,word_counts,warning_timestamps)
    N_nonempty_lines += N1
    N_error_files += N2

top_words = find_most_frequent_words(word_counts)
t_max = find_longest_period(warning_timestamps)

# output
print(len(file_paths))
print(N_nonempty_lines)
print(N_error_files)
print((', '.join(top_words)))
print(t_max)
