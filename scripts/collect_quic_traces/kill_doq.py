import os, signal
# iterating through each instance of the process
for line in os.popen("ps ax | grep dnsproxy | grep -v grep"):
    fields = line.split()

    # extracting Process ID from the output
    pid = fields[0]

    # terminating process
    os.kill(int(pid), signal.SIGKILL)
