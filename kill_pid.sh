# Function to check and kill processes on a given port
port = $(netstat -ano |grep 8080 )

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
# Add a small delay and retry mechanism
for i in {1..3}; do
    if rm -f /c/Users/Administrator/AppData/Local/Temp/.model_server.pid 2>/dev/null; then

        echo 'PID file removed'
        break
    else
        echo "Attempt $i: Failed to remove PID file, waiting 2 seconds..."
        sleep 2
    fi
done