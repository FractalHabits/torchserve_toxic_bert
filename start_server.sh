#!/bin/bash

# Kill any existing torchserve processes
echo 'Killing any existing torchserve processes...'
taskkill /F /IM "torchserve.exe" /T 2>/dev/null || true
echo 'Torchserve processes killed'

# Remove PID file if it exists
echo 'Removing PID file...'
rm -f /c/Users/User/AppData/Local/Temp/.model_server.pid
echo 'PID file removed'

# Create model store directory if it doesn't exist
echo 'Creating model store directory...'
mkdir -p model_store
echo 'Model store directory created'

# Archive the model with force flag
echo 'Archiving model...'
torch-model-archiver --model-name toxic_bert \
                    --version 1.0 \
                    --model-file toxic_bert.pth \
                    --handler text_handler.py \
                    --extra-files "model_definition.py, tokenizer.py" \
                    --export-path model_store \
                    --force > /dev/null 2>&1  # Redirect output to /dev/null
                    #--force > archiver_output.log 2>&1 
echo 'Model archived'

# Start TorchServe with config
echo 'Starting TorchServe with config...'
torchserve --start \
           --model-store model_store \
           --models toxic_bert=toxic_bert.mar \
           --disable-token-auth \
           --ts-config config.properties \
           --ncs > torchserve_output.log 2>&1 & # Log output to a file
            ##--force > /dev/null 2>&1  # Redirect both stdout and stderr to /dev/null    
# Check if the log file was created
if [ ! -f torchserve_output.log ]; then
    echo "Log file not created. There may have been an error starting TorchServe."
else
    # Check the health of TorchServe
    echo 'Checking TorchServe health...'
    curl -s http://127.0.0.1:8081/ping  # Check if the server is running

    if [ $? -eq 0 ]; then
        echo "TorchServe started successfully. Use Ctrl+C to stop the server."
    else
        echo "TorchServe failed to start. Check the logs for more details."
        #echo "TorchServe logs:"
        #cat torchserve_output.log  # Display the log file contents
    fi
fi