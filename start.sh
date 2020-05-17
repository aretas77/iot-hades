#!/bin/sh

testMac=00:00:00:00:00:00

# create files for testing
touch models/"${testMac}"
cat > states/"${testMac}" << EOL
{"stats": {"prev_temperature": 25.2, "prev_delta": 0.5, "curr_temperature": 25.2, "send_interval": 2.0}}
EOL

python hades.py --enable_select_tf_ops
