#!/usr/bin/expect
spawn ./sendKernelToServer.sh
expect "user@10.10.1.149's password:"
send "password\r";
interact