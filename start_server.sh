#!/bin/bash

# Function to check and kill processes on a given port
function kill_process_on_port {
    # Declare a local variable 'port' to hold the port number passed to the function
    local port=$1
    echo "Checking port $port..."

    # Use 'netstat' to list all network connections and filter for the specified port
    # 'grep' is used to find lines containing the port number
    # 'awk' extracts the fifth column, which is the PID of the process using the port
    local pid=$(netstat -ano | grep ":$port" | awk '{print $5}')

    # Check if a PID was found (i.e., if the variable 'pid' is not empty)
    if [ -n "$pid" ]; then
        echo "Port $port is used by PID $pid. Terminating process..."
        # Use 'taskkill' to forcefully terminate the process with the found PID
        # '//F' forces the termination, and '//PID' specifies the process ID
        taskkill //F //PID $pid
    else
        # If no PID was found, print a message indicating no process is using the port
        echo "No process is using port $port."
    fi
}

# Loop over the list of ports: 8080, 8081, and 8082
for port in 8080 8081 8082; do
    # Call the function 'kill_process_on_port' with the current port number
    kill_process_on_port $port
done

# Remove PID file if it exists
echo 'Removing PID file...'
rm -f /c/Users/User/AppData/Local/Temp/.model_server.pid
echo 'PID file removed'

# Archive the model with force flag
echo 'Archiving model...'
torch-model-archiver --model-name toxic_bert \
                    --version 1.0 \
                    --model-file toxic_bert.pth \
                    --handler text_handler.py \
                    --export-path model_store \
                    --force
echo 'Model archived'

echo 'Starting TorchServe with config...'
torchserve --start \
           --model-store model_store \
           --models toxic_bert=toxic_bert.mar \
           --disable-token-auth \
           --ts-config config.properties \
           --ncs #> torchserve_output.log 2>&1 & # Log output to a file
